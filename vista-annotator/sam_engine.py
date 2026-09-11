"""SAM3 tracker engine: box-prompted mask propagation over a chunk of video frames.

Loads the SAM3 tracker (SAM2-style promptable video tracker + SAM3 vision backbone)
once, lazily, on the GPU with the most free memory. Propagation jobs run one at a
time (GPU lock) in a background thread; progress is polled via Job objects.
"""
import os
import shutil
import tempfile
import threading
import traceback
import uuid

import numpy as np

CKPT = os.path.expanduser(
    "~/.cache/huggingface/hub/models--facebook--sam3/snapshots/"
    "3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt")

_model = None
_model_lock = threading.Lock()
_gpu_lock = threading.Lock()


def _pick_device():
    import torch
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return f"cuda:{best}"


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            import torch
            from sam3.model_builder import build_tracker
            dev = _pick_device()
            m = build_tracker(apply_temporal_disambiguation=True, with_backbone=True)
            ck = torch.load(CKPT, map_location="cpu", weights_only=True)
            if "model" in ck and isinstance(ck["model"], dict):
                ck = ck["model"]
            remap = {}
            for k, v in ck.items():
                if k.startswith("tracker."):
                    remap[k[len("tracker."):]] = v
                elif k.startswith("detector.backbone."):
                    remap["backbone." + k[len("detector.backbone."):]] = v
            missing, _ = m.load_state_dict(remap, strict=False)
            assert not missing, f"missing keys: {missing[:5]}"
            m = m.to(dev).eval()  # m.device is a property derived from parameters
            _model = m
        return _model


def _iou(a, b):
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0.0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0.0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


class Job:
    def __init__(self, video_id, start, end, boxes):
        self.id = uuid.uuid4().hex[:12]
        self.video_id = video_id
        self.start = start
        self.end = end
        self.boxes = boxes          # [{ann_id, category_id, bbox}]
        self.status = "queued"      # queued | extracting | tracking | done | error | cancelled
        self.progress = 0.0
        self.current_frame = start
        self.message = ""
        self.created = []           # ann dicts added to the store
        self.lost = {}              # track_id -> frame where lost
        self.cancel_flag = False

    def as_dict(self):
        return {"id": self.id, "video_id": self.video_id, "status": self.status,
                "progress": round(self.progress, 3), "current_frame": self.current_frame,
                "start": self.start, "end": self.end, "message": self.message,
                "n_created": len(self.created), "lost": self.lost}


def _extract_frames(video_path, start, end, out_dir, job):
    """Dump frames [start, end] as JPEGs named 00000.jpg... using decord."""
    import cv2
    from decord import VideoReader
    vr = VideoReader(video_path, num_threads=2)
    n = end - start + 1
    for i in range(n):
        if job.cancel_flag:
            return False
        frame = vr[start + i].asnumpy()  # RGB
        cv2.imwrite(os.path.join(out_dir, f"{i:05d}.jpg"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        job.progress = 0.25 * (i + 1) / n
    return True


def run_propagation(job, video_path, store, min_area=12.0, save_cb=None):
    """Blocking worker. Adds boxes at job.start, propagates to job.end,
    converts masks to bboxes, writes into `store` frame by frame."""
    import torch
    try:
        with _gpu_lock:
            job.status = "extracting"
            model = get_model()
            tmp = tempfile.mkdtemp(prefix="sam3_chunk_", dir="/media/ssd/pasquale/tmp")
            try:
                if not _extract_frames(video_path, job.start, job.end, tmp, job):
                    job.status = "cancelled"
                    return
                job.status = "tracking"
                W, H = store.width, store.height
                state = model.init_state(video_path=tmp, offload_video_to_cpu=True)
                obj_meta = {}
                for i, b in enumerate(job.boxes, 1):
                    x, y, w, h = b["bbox"]
                    box = np.array([x / W, y / H, (x + w) / W, (y + h) / H],
                                   dtype=np.float32)
                    model.add_new_points_or_box(state, frame_idx=0, obj_id=i,
                                                box=box, rel_coordinates=True)
                    obj_meta[i] = b
                n = job.end - job.start + 1
                existing = store.annotated_frames()
                with torch.inference_mode():
                    for fidx, obj_ids, _low, video_res_masks, obj_scores in \
                            model.propagate_in_video(
                                state, start_frame_idx=0,
                                max_frame_num_to_track=n, reverse=False,
                                tqdm_disable=True, propagate_preflight=True):
                        if job.cancel_flag:
                            job.status = "cancelled"
                            return
                        abs_frame = job.start + fidx
                        job.current_frame = abs_frame
                        job.progress = 0.25 + 0.75 * (fidx + 1) / n
                        if fidx == 0:
                            continue  # prompt frame already annotated
                        masks = (video_res_masks > 0.0).squeeze(1).cpu().numpy()
                        scores = obj_scores.reshape(-1).float().cpu().numpy()
                        for oid, m, sc in zip(obj_ids, masks, scores):
                            oid = int(oid)
                            if oid in job.lost:
                                continue
                            ys, xs = np.nonzero(m)
                            if len(xs) == 0 or len(xs) < min_area or sc < 0:
                                job.lost[oid] = abs_frame
                                continue
                            x0, x1 = xs.min(), xs.max()
                            y0, y1 = ys.min(), ys.max()
                            # mask is at model resolution == video resolution here
                            # (propagate returns masks resized to video H,W)
                            bbox = [float(x0), float(y0),
                                    float(x1 - x0 + 1), float(y1 - y0 + 1)]
                            meta = obj_meta[oid]
                            # skip if an equivalent box already exists on this frame
                            if any(_iou(bbox, e["bbox"]) > 0.5
                                   for e in store.get_range(abs_frame, abs_frame)
                                   .get(abs_frame, [])):
                                continue
                            # caption mode: one tracked object may carry extra
                            # caption categories -> duplicate bbox per category
                            cats = [meta["category_id"]] + meta.get("extra_cats", [])
                            for cat in cats:
                                ann = store.add(abs_frame, cat, bbox,
                                                source="sam3", run_id=job.id,
                                                track_id=oid)
                                job.created.append(ann)
                        if save_cb and fidx % 150 == 0:
                            save_cb()
                del state
                torch.cuda.empty_cache()
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if save_cb:
            save_cb()
        job.progress = 1.0
        job.status = "done"
        job.message = f"{len(job.created)} boxes created on frames {job.start + 1}-{job.current_frame}"
    except Exception as e:
        traceback.print_exc()
        job.status = "error"
        job.message = f"{type(e).__name__}: {e}"
