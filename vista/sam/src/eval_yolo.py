"""
Evaluation utilities for YOLO-style object detection predictions.

- Input format (YOLO txt):
    class cx cy w h [score]

- Coordinates are normalized in [0, 1]; IoU is computed in this space, so
  image size is not needed.

- Metrics computed:
    * AP@0.50 per class
    * AP@0.50:0.95 per class (average over IoU thresholds 0.50..0.95 step 0.05)
    * Precision / Recall / F1 per class at a fixed score threshold
    * Global (micro) Precision / Recall / F1
    * mAP@0.50 and mAP@0.50:0.95 over all classes
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np


@dataclass
class ClassMetrics:
    num_gt: int
    num_pred: int
    ap_50: float
    ap_50_95: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int


@dataclass
class EvaluationResult:
    per_class: Dict[int, ClassMetrics]
    map_50: float
    map_50_95: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    total_gt: int
    total_pred: int


# ---------------------------------------------------------------------#
# Core helpers
# ---------------------------------------------------------------------#


def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float) -> np.ndarray:
    """Convert YOLO (cx, cy, w, h) in [0,1] to (x1, y1, x2, y2)."""
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    x2 = cx + w / 2.0
    y2 = cy + h / 2.0
    return np.asarray([x1, y1, x2, y2], dtype=np.float32)


def _compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """
    Compute IoU between a single box and an array of boxes.

    box:   (4,)  -> [x1, y1, x2, y2]
    boxes: (N,4)
    """
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)

    box = box.reshape(1, 4)

    x1 = np.maximum(box[:, 0], boxes[:, 0])
    y1 = np.maximum(box[:, 1], boxes[:, 1])
    x2 = np.minimum(box[:, 2], boxes[:, 2])
    y2 = np.minimum(box[:, 3], boxes[:, 3])

    inter_w = np.clip(x2 - x1, 0.0, None)
    inter_h = np.clip(y2 - y1, 0.0, None)
    inter = inter_w * inter_h

    area_box = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_box + area_boxes - inter

    return inter / np.clip(union, 1e-16, None)


def _voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """
    Compute AP using the 11-point interpolation (VOC 2007 style).

    This is sufficient for a consistent comparison with earlier work
    and matches the typical "mAP@0.50" definition for small projects.
    """
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 11):
        if np.any(rec >= t):
            p = float(np.max(prec[rec >= t]))
        else:
            p = 0.0
        ap += p / 11.0
    return ap


# ---------------------------------------------------------------------#
# Loading ground-truth and predictions
# ---------------------------------------------------------------------#


def _load_yolo_dataset(
    labels_dir: Path,
    preds_dir: Path,
    num_classes: int,
    confidence_threshold: float,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, List[Tuple[str, np.ndarray, float]]]]:
    """
    Load YOLO-style labels and predictions.

    Returns:
        gt_by_class:   dict[class_id][image_id] -> (N_gt, 4) array
        preds_by_class: dict[class_id] -> list of (image_id, box_xyxy, score)
    """
    gt_by_class: Dict[int, Dict[str, List[List[float]]]] = {c: {} for c in range(num_classes)}
    preds_by_class: Dict[int, List[Tuple[str, np.ndarray, float]]] = {
        c: [] for c in range(num_classes)
    }

    # Ground-truth labels
    for label_path in sorted(labels_dir.glob("*.txt"), key=lambda p: int(p.stem)):
        image_id = label_path.stem
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    continue
                cx, cy, w, h = map(float, parts[1:5])
                box = _yolo_to_xyxy(cx, cy, w, h).tolist()
                gt_by_class[class_id].setdefault(image_id, []).append(box)

    gt_by_class_arr: Dict[int, Dict[str, np.ndarray]] = {c: {} for c in range(num_classes)}
    for c in range(num_classes):
        for image_id, boxes in gt_by_class[c].items():
            gt_by_class_arr[c][image_id] = np.asarray(boxes, dtype=np.float32)

    # Predictions
    for pred_path in sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem)):
        image_id = pred_path.stem
        with pred_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    continue
                cx, cy, w, h = map(float, parts[1:5])
                score = float(parts[5]) if len(parts) >= 6 else 1.0
                if score < confidence_threshold:
                    continue
                box = _yolo_to_xyxy(cx, cy, w, h)
                preds_by_class[class_id].append((image_id, box, score))

    return gt_by_class_arr, preds_by_class


# ---------------------------------------------------------------------#
# Evaluation logic
# ---------------------------------------------------------------------#


def _evaluate_for_iou(
    gt_by_class: Dict[int, Dict[str, np.ndarray]],
    preds_by_class: Dict[int, List[Tuple[str, np.ndarray, float]]],
    num_classes: int,
    iou_threshold: float,
) -> Tuple[Dict[int, float], Dict[int, Tuple[int, int, int]]]:
    """
    Evaluate predictions for a single IoU threshold.

    Returns:
        ap_per_class: dict[class_id] -> AP at this IoU
        pr_stats:     dict[class_id] -> (TP_total, FP_total, num_gt)
                      (TP/FP are meaningful primarily for the lowest IoU, e.g. 0.5)
    """
    ap_per_class: Dict[int, float] = {}
    pr_stats: Dict[int, Tuple[int, int, int]] = {}

    for c in range(num_classes):
        gt_dict = gt_by_class.get(c, {})
        preds = preds_by_class.get(c, [])

        npos = sum(len(v) for v in gt_dict.values())
        if npos == 0 and len(preds) == 0:
            ap_per_class[c] = 0.0
            pr_stats[c] = (0, 0, 0)
            continue

        preds_sorted = sorted(preds, key=lambda x: x[2], reverse=True)
        tp = np.zeros(len(preds_sorted), dtype=np.float32)
        fp = np.zeros(len(preds_sorted), dtype=np.float32)

        matched: Dict[str, np.ndarray] = {}
        for img_id, boxes in gt_dict.items():
            matched[img_id] = np.zeros(len(boxes), dtype=bool)

        for i, (img_id, box_pred, _score) in enumerate(preds_sorted):
            gt_boxes = gt_dict.get(img_id)
            if gt_boxes is None or gt_boxes.size == 0:
                fp[i] = 1.0
                continue

            ious = _compute_iou(box_pred, gt_boxes)
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_threshold and not matched[img_id][best_idx]:
                tp[i] = 1.0
                matched[img_id][best_idx] = True
            else:
                fp[i] = 1.0

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        if npos == 0:
            rec = np.zeros_like(tp_cum)
        else:
            rec = tp_cum / float(npos)
        prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-16)

        if npos == 0 or tp_cum.size == 0:
            ap = 0.0
            TP_total = FP_total = 0
        else:
            ap = _voc_ap(rec, prec)
            TP_total = int(tp_cum[-1])
            FP_total = int(fp_cum[-1])

        ap_per_class[c] = float(ap)
        pr_stats[c] = (TP_total, FP_total, npos)

    return ap_per_class, pr_stats


def evaluate_yolo_predictions(
    labels_dir: Path,
    preds_dir: Path,
    num_classes: int,
    confidence_threshold: float = 0.0,
) -> EvaluationResult:
    """
    Evaluate YOLO-style predictions against ground-truth annotations.

    Args:
        labels_dir: directory with YOLO GT files (*.txt).
        preds_dir:  directory with YOLO prediction files (*.txt).
        num_classes: number of classes (0..num_classes-1).
        confidence_threshold: discard predictions with score < threshold.

    Returns:
        EvaluationResult with AP, mAP and P/R/F1 metrics.
    """
    gt_by_class, preds_by_class = _load_yolo_dataset(
        labels_dir=labels_dir,
        preds_dir=preds_dir,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    )

    # AP@0.50 and P/R/F1 at IoU=0.50
    ap_50_per_class, pr_stats_50 = _evaluate_for_iou(
        gt_by_class, preds_by_class, num_classes, iou_threshold=0.5
    )

    # AP@0.50:0.95
    iou_thresholds = [0.5 + 0.05 * i for i in range(10)]  # 0.50 .. 0.95
    ap_per_iou: Dict[float, Dict[int, float]] = {}
    for thr in iou_thresholds:
        ap_thr, _ = _evaluate_for_iou(gt_by_class, preds_by_class, num_classes, thr)
        ap_per_iou[thr] = ap_thr

    ap_50_95_per_class: Dict[int, float] = {}
    for c in range(num_classes):
        vals = [ap_per_iou[thr][c] for thr in iou_thresholds]
        ap_50_95_per_class[c] = float(np.mean(vals)) if vals else 0.0

    per_class: Dict[int, ClassMetrics] = {}
    total_TP = 0
    total_FP = 0
    total_gt = 0

    for c in range(num_classes):
        TP, FP, npos = pr_stats_50[c]
        total_TP += TP
        total_FP += FP
        total_gt += npos

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / npos if npos > 0 else 0.0
        if precision + recall > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        per_class[c] = ClassMetrics(
            num_gt=npos,
            num_pred=TP + FP,
            ap_50=ap_50_per_class[c],
            ap_50_95=ap_50_95_per_class[c],
            precision=precision,
            recall=recall,
            f1=f1,
            tp=TP,
            fp=FP,
        )

    if total_TP + total_FP > 0:
        micro_precision = total_TP / (total_TP + total_FP)
    else:
        micro_precision = 0.0

    micro_recall = total_TP / total_gt if total_gt > 0 else 0.0
    if micro_precision + micro_recall > 0:
        micro_f1 = 2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = 0.0

    map_50 = float(np.mean([per_class[c].ap_50 for c in range(num_classes)]))
    map_50_95 = float(np.mean([per_class[c].ap_50_95 for c in range(num_classes)]))

    return EvaluationResult(
        per_class=per_class,
        map_50=map_50,
        map_50_95=map_50_95,
        micro_precision=micro_precision,
        micro_recall=micro_recall,
        micro_f1=micro_f1,
        total_gt=total_gt,
        total_pred=total_TP + total_FP,
    )


def print_evaluation_summary(
    result: EvaluationResult,
    class_names: Optional[Dict[int, str]] = None,
) -> None:
    """Pretty-print evaluation metrics."""
    print(
        f"Loaded {result.total_gt} ground-truth boxes and "
        f"{result.total_pred} predictions."
    )
    print("\nPer-class metrics (IoU >= 0.5, fixed score threshold):\n")

    for class_id, metrics in sorted(result.per_class.items()):
        name = class_names.get(class_id, str(class_id)) if class_names else str(class_id)
        print(
            f"  Class {class_id} ({name}):\n"
            f"    GT={metrics.num_gt}, Pred={metrics.num_pred}, "
            f"TP={metrics.tp}, FP={metrics.fp}\n"
            f"    P = {metrics.precision:.4f}, R = {metrics.recall:.4f}, "
            f"F1 = {metrics.f1:.4f}\n"
            f"    AP@0.50     = {metrics.ap_50:.4f}\n"
            f"    AP@0.50:0.95 = {metrics.ap_50_95:.4f}\n"
        )

    print("Global metrics:\n")
    print(
        f"  P = {result.micro_precision:.4f}, "
        f"R = {result.micro_recall:.4f}, "
        f"F1 = {result.micro_f1:.4f}\n"
        f"  mAP@0.50     = {result.map_50:.4f}\n"
        f"  mAP@0.50:0.95 = {result.map_50_95:.4f}"
    )
