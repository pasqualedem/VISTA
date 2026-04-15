"""
Utility functions to convert SAM 3 predictions into YOLO-style bounding boxes,
plus a simple class-wise NMS to reduce duplicates (YOLO-like postprocess).

The main steps are:
- take the pixel-space bounding boxes produced by SAM 3,
- convert them to the normalized YOLO format (cx, cy, w, h in [0,1]),
- optionally apply Non-Maximum Suppression to remove duplicate boxes,
- serialize the result into text lines ready to be written to .txt files.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .sam3_wrapper import Sam3Prediction


@dataclass
class YoloBox:
    """YOLO-format bounding box, optionally with a confidence score."""
    class_id: int
    cx: float
    cy: float
    w: float
    h: float
    score: Optional[float] = None
    features: Optional[np.ndarray] = None  # Query embeddings (256-d) from SAM3


def sam3_boxes_to_yolo(
    prediction: Sam3Prediction,
    class_id: int,
    image_width: int,
    image_height: int,
    score_threshold: float = 0.0,
) -> List[YoloBox]:
    """
    Convert SAM 3 pixel-space boxes into YOLO-normalized boxes for a single class.

    SAM 3 boxes are expected as [x1, y1, x2, y2] in pixel coordinates.
    Output: YOLO format (cx, cy, w, h) normalized to [0, 1].
    """
    # Extract raw boxes, scores, and features from the SAM 3 prediction.
    boxes = prediction.boxes
    scores = prediction.scores
    features = prediction.features

    # If they are tensors (PyTorch, etc.), convert them to NumPy arrays.
    if hasattr(boxes, "detach"):
        boxes_np = boxes.detach().cpu().numpy()
    else:
        boxes_np = np.asarray(boxes)

    if hasattr(scores, "detach"):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    # Convert features to numpy if present
    features_np = None
    if features is not None:
        if hasattr(features, "detach"):
            features_np = features.detach().cpu().numpy()
        else:
            features_np = np.asarray(features)

    yolo_boxes: List[YoloBox] = []
    W = float(image_width)
    H = float(image_height)

    # If there are no predictions, return an empty list.
    if boxes_np.size == 0 or scores_np.size == 0:
        return yolo_boxes

    # Iterate over each box+score+feature proposed by SAM 3 for this class.
    num_predictions = len(boxes_np)
    for idx in range(num_predictions):
        box = boxes_np[idx]
        score = scores_np[idx]
        feat = features_np[idx] if features_np is not None else None
        score_f = float(score)
        if score_f < score_threshold:
            # Discard predictions with too low confidence.
            continue

        # --- clip + validate: avoid exporting boxes outside the image or degenerate ones ---
        x1, y1, x2, y2 = map(float, box)

        # Clip to image bounds (pixel coordinates).
        x1 = max(0.0, min(W, x1))
        x2 = max(0.0, min(W, x2))
        y1 = max(0.0, min(H, y1))
        y2 = max(0.0, min(H, y2))

        # If after clipping the box is empty or degenerate, skip it.
        if x2 <= x1 or y2 <= y1:
            continue
        # --- end clip + validate ---

        # Convert from corner coordinates (x1,y1,x2,y2) to (cx,cy,w,h) in pixels.
        bw = x2 - x1
        bh = y2 - y1
        cx = x1 + bw / 2.0
        cy = y1 + bh / 2.0

        # Normalize by image size to obtain values in [0,1].
        cx_norm = cx / W
        cy_norm = cy / H
        bw_norm = bw / W
        bh_norm = bh / H

        # Safety clamp to limit small numerical errors outside [0,1].
        cx_norm = float(np.clip(cx_norm, 0.0, 1.0))
        cy_norm = float(np.clip(cy_norm, 0.0, 1.0))
        bw_norm = float(np.clip(bw_norm, 0.0, 1.0))
        bh_norm = float(np.clip(bh_norm, 0.0, 1.0))

        # If desired, we could also discard boxes with almost-zero w/h:
        # if bw_norm <= 0.0 or bh_norm <= 0.0:
        #     continue

        # Append the converted box to the list in YOLO format.
        yolo_boxes.append(
            YoloBox(
                class_id=class_id,
                cx=cx_norm,
                cy=cy_norm,
                w=bw_norm,
                h=bh_norm,
                score=score_f,
                features=feat,
            )
        )

    return yolo_boxes


def yolo_boxes_to_lines(
    boxes: List[YoloBox],
    include_score_column: bool = False,
    include_score_comment: bool = False,
) -> List[str]:
    """
    Convert a list of YoloBox objects into YOLO-style text lines.

    If include_score_column is True, the confidence is written as a 6th numeric
    column: class cx cy w h score.

    If include_score_comment is True, the confidence is appended as a comment.
    """
    lines: List[str] = []

    # Each YoloBox is converted into a single text line.
    for box in boxes:
        base = f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f}"

        if include_score_column and box.score is not None:
            # Add the score as a sixth numeric column.
            base = f"{base} {box.score:.6f}"
        elif include_score_comment and box.score is not None:
            # Or append the score as a comment (not used for automatic parsing).
            base = f"{base}  # score={box.score:.3f}"

        lines.append(base)

    return lines


# ---------------------------------------------------------------------
# NMS (class-wise) for YOLO boxes (normalized coords)
# ---------------------------------------------------------------------


def _yolo_to_xyxy_norm(b: YoloBox) -> np.ndarray:
    """Convert YOLO (cx,cy,w,h) normalized to xyxy normalized."""
    # Convert from (center, width, height) representation
    # to corner coordinates (x1,y1,x2,y2), still normalized.
    x1 = b.cx - b.w / 2.0
    y1 = b.cy - b.h / 2.0
    x2 = b.cx + b.w / 2.0
    y2 = b.cy + b.h / 2.0
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _iou_xyxy(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """IoU between one xyxy box a (4,) and array B (N,4). Coordinates normalized."""
    if B.size == 0:
        return np.zeros((0,), dtype=np.float32)

    # Compute intersection between box a and all boxes in B.
    x1 = np.maximum(a[0], B[:, 0])
    y1 = np.maximum(a[1], B[:, 1])
    x2 = np.minimum(a[2], B[:, 2])
    y2 = np.minimum(a[3], B[:, 3])

    inter = np.clip(x2 - x1, 0.0, None) * np.clip(y2 - y1, 0.0, None)
    area_a = np.clip(a[2] - a[0], 0.0, None) * np.clip(a[3] - a[1], 0.0, None)
    area_B = np.clip(B[:, 2] - B[:, 0], 0.0, None) * np.clip(B[:, 3] - B[:, 1], 0.0, None)
    union = area_a + area_B - inter
    return inter / np.clip(union, 1e-16, None)


def nms_yolo_boxes(
    boxes: List[YoloBox],
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> List[YoloBox]:
    """
    Class-wise greedy NMS over YOLO-normalized boxes.

    - Suppresses duplicates within the SAME class.
    - Keeps higher-score boxes, removes boxes with IoU > iou_threshold.
    """
    # Group boxes by class_id: NMS is applied separately per class.
    by_class: Dict[int, List[YoloBox]] = {}
    for b in boxes:
        by_class.setdefault(b.class_id, []).append(b)

    kept_all: List[YoloBox] = []

    for cls, cls_boxes in by_class.items():
        # Require that boxes have a score (needed to sort them).
        cls_boxes = [b for b in cls_boxes if b.score is not None]
        if not cls_boxes:
            continue

        # Sort class boxes by descending score (greedy NMS).
        cls_boxes.sort(key=lambda b: float(b.score), reverse=True)

        # Precompute normalized xyxy coordinates and scores as NumPy arrays.
        xyxy = np.stack([_yolo_to_xyxy_norm(b) for b in cls_boxes], axis=0)
        scores = np.asarray([float(b.score) for b in cls_boxes], dtype=np.float32)

        order = np.argsort(-scores)
        keep_idx: List[int] = []

        # Main greedy NMS loop: keep the best-scoring box and discard overlapping ones.
        while order.size > 0 and len(keep_idx) < max_det:
            i = int(order[0])
            keep_idx.append(i)
            if order.size == 1:
                break

            rest = order[1:]
            ious = _iou_xyxy(xyxy[i], xyxy[rest])
            # Keep only boxes whose IoU with the chosen one is <= threshold.
            order = rest[ious <= iou_threshold]

        kept_all.extend([cls_boxes[i] for i in keep_idx])

    # Global cap + sort by score: sort all kept boxes by score and apply a global limit.
    kept_all.sort(key=lambda b: float(b.score or 0.0), reverse=True)
    return kept_all[:max_det]
