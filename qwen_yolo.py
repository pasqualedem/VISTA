import base64
from collections import deque
from io import BytesIO
import os
import cv2
import json
import yaml
import torch
import gc
from pathlib import Path
from typing import Dict

from PIL import Image
from ultralytics import YOLO
from json_repair import repair_json
from vista.qwen import get_model
from vista.utils import set_seed, image_to_base64, resize_image, log, IGNORE_CATEGORIES
from vista.utils import get_emergency_level


os.environ["HF_HOME"] = "/media/nvme/pasquale/HF"


# ============================================================
# Geometry
# ============================================================

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = (a[2] - a[0]) * (a[3] - a[1])
    areaB = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (areaA + areaB - inter)

# ============================================================
# Drawing
# ============================================================

def draw_tracks(frame, tracks):
    for tid, t in tracks.items():
        # Assign a color based on the emergency level
        color_db = {
            3: (0, 0, 255),    # Red for high emergency
            2: (0, 165, 255),  # Orange for medium emergency
            1: (0, 255, 255),  # Yellow for low emergency
            0: (255, 0, 0),    # Blue for no emergency
        }
        color = color_db.get(t.get("emergency_level", 0), (255, 0, 0))  # Default to blue if not found
        x1, y1, x2, y2 = map(int, t["bbox"])
        label = t.get("label", "unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        # draw white rectangle background for text
        (text_w, text_h), _ = cv2.getTextSize(f"{tid}:{label}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, max(0, y1 - 6 - text_h)), (x1 + text_w, max(0, y1 - 6 + 2)), (0,0,0), -1)
        cv2.putText(
            frame,
            f"{tid}:{label}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
    return frame

def draw_qwen_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["bbox_2d"])
        label = det.get("label", "unknown")
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            label,
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )
    return frame


def postprocess_boxes(detections: list, image: Image.Image, model_id: str) -> list:
    scale = not model_id.startswith("Qwen/Qwen2.5-VL")
    if not scale:
        return detections

    x_scale = image.width / 1000
    y_scale = image.height / 1000

    out = []
    for d in detections:
        xmin, ymin, xmax, ymax = d["bbox_2d"]
        d["bbox_2d"] = [
            int(xmin * x_scale),
            int(ymin * y_scale),
            int(xmax * x_scale),
            int(ymax * y_scale),
        ]
        out.append(d)
    return out


# ============================================================
# Pipeline
# ============================================================

def run_pipeline(cfg: dict):
    video_path = cfg["input"]["video"]
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    device = cfg["device"]

    log("Loading YOLO tracker")
    yolo = YOLO(cfg["yolo"]["model"])

    
    load_prediction_dir = cfg["qwen"].get("load_prediction_dir", None)
    
    if load_prediction_dir is not None:
        model = None
        log(f"Loading Qwen predictions from {load_prediction_dir}, skipping Qwen inference")
    else:
        log("Loading Qwen processor and model")
        model = get_model(cfg)
        log("Models loaded successfully")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
    writer = cv2.VideoWriter(
        str(out_dir / "annotated.mp4"),
        fourcc,
        fps,
        (W, H),
    )

    track_db: Dict[int, Dict] = {}
    frame_idx = 0

    stride = cfg["qwen"]["every_n_frames"]
    iou_thr = cfg["yolo"]["iou_match_threshold"]
    qwen_N = cfg["qwen"].get("num_frames", 1)  # how many frames to give to Qwen

    (out_dir / "qwen").mkdir(exist_ok=True)
    qwen_history = deque(maxlen=qwen_N)

    log("Starting video loop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo.track(frame, persist=True, verbose=False, conf=cfg["yolo"].get("conf", None))[0]
        active_tracks = {}

        if results.boxes.id is not None:
            for box, tid, cls in zip(results.boxes.xyxy, results.boxes.id, results.boxes.cls):
                tid = int(tid.item())
                active_tracks[tid] = {
                    "bbox": box.cpu().numpy().tolist(),
                    "label": track_db.get(tid, {}).get("label", results.names.get(int(cls.item()), "unknown")),
                    "emergency_level": track_db.get(tid, {}).get("emergency_level", 0),
                }
        # Remove inactive tracks from track_db
        inactive_tids = set(track_db.keys()) - set(active_tracks.keys())
        for tid in inactive_tids:
            del track_db[tid]
        
        # Remove tracks belonging to ignored categories
        active_tracks = {tid: tr for tid, tr in active_tracks.items() if tr["label"] not in IGNORE_CATEGORIES}

        # ----------------------------
        # Update frame memory
        # ----------------------------
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # ----------------------------
        # Qwen inference
        # ----------------------------
        if frame_idx % stride == 0:
            log(f"Qwen inference at frame {frame_idx}")

            # Run Qwen
            if load_prediction_dir is not None:
                raw_path = Path(load_prediction_dir) / "qwen" / f"{frame_idx:06d}_raw.txt"
                raw = raw_path.read_text()
                log(f"Loaded Qwen predictions from {raw_path}")
            else:
                raw = model.generate(frame=pil_frame, history=list(qwen_history))
                
            qwen_history.append((pil_frame, raw))

            qwen_dir = out_dir / "qwen"
            raw_path = qwen_dir / f"{frame_idx:06d}_raw.txt"
            raw_path.write_text(raw)

            repaired = repair_json(raw)
            (qwen_dir / f"{frame_idx:06d}_repaired.json").write_text(repaired)

            try:
                parsed = json.loads(repaired)
                parsed = postprocess_boxes(parsed, pil_frame, cfg["qwen"]["model_id"])
                
                qwen_annotated = draw_qwen_detections(frame.copy(), parsed)
                cv2.imwrite(
                    str(qwen_dir / f"{frame_idx:06d}_annotated.png"),
                    qwen_annotated,
                )
            except Exception as e:
                log(f"Qwen JSON parse error at frame {frame_idx}: {e}")
                parsed = []

            # ----------------------------
            # Match Qwen detections to tracks
            # ----------------------------
            matches = 0
            for det in parsed:
                qb = det["bbox_2d"]
                label = det.get("label", "unknown")
                urgency_level = det.get("level", get_emergency_level(label))

                best_tid = None
                best_score = 0.0
                for tid, tr in active_tracks.items():
                    score = iou(qb, tr["bbox"])
                    if score > best_score:
                        best_score = score
                        best_tid = tid

                if best_tid is not None and best_score >= iou_thr:
                    track_db[best_tid] = {
                        "bbox": active_tracks[best_tid]["bbox"],
                        "label": label,
                        "emergency_level": urgency_level,
                    }
                    matches += 1
            log(f"YOLO detected {len(active_tracks)} tracks, Qwen detected {len(parsed)} objects, matched {matches}")

        # ----------------------------
        # Update track DB
        # ----------------------------
        for tid, tr in active_tracks.items():
            if tid not in track_db:
                track_db[tid] = tr
            else:
                track_db[tid]["bbox"] = tr["bbox"]

        annotated = draw_tracks(frame.copy(), track_db)
        if cfg.get("save_annotated_frames_frequency", 0) > 0 and frame_idx % cfg["save_annotated_frames_frequency"] == 0:
            Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)).save(
                out_dir / f"annotated_frame_{frame_idx:06d}.png"
            )
        writer.write(annotated)
        frame_idx += 1

# ============================================================
# Entry point
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
