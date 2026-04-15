from __future__ import annotations

"""
Build a training dataset for a linear probe on top of SAM 3 detections.

For a given split (e.g. "train"), this script:
- Loads YOLO-style ground-truth boxes.
- Loads SAM 3 YOLO-style predictions for the same split.
- Loads semantic features (257-d: 256-d query embeddings + score) from .npz files.
- Matches predictions to ground truth with IoU >= 0.5 (same logic as eval_yolo).
- For each prediction, uses the 257-d semantic feature vector aligned by line index.
- Assigns a binary label:
    1 = true positive (TP), 0 = false positive (FP) according to the matching.
- Saves the resulting features and labels to:
    data/processed/linear_probe/sam3_linear_probe_<split>.npz
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

# Add project root to PYTHONPATH so that "src" can be imported when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import get_labels_dir, get_sam3_yolo_predictions_dir  # noqa: E402
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.eval_yolo import (  # noqa: E402
    _load_yolo_dataset,
    _compute_iou,
)


def _sort_key(p: Path):
    """
    Safe sorting key that tries numeric sorting first, falls back to lexicographic.
    
    This prevents crashes when file stems are not purely numeric (e.g., 'img_559')
    while maintaining numeric sorting when possible (e.g., '100' comes after '99').
    """
    try:
        return int(p.stem)
    except ValueError:
        return p.stem


def _load_features_for_image(
    features_dir: Path,
    image_id: str,
) -> Optional[np.ndarray]:
    """
    Load the query features saved for an image.

    Args:
        features_dir: Directory containing .npz feature files
        image_id: Image identifier (stem of the image file)

    Returns:
        Features array of shape (N, 257) where N is number of predictions.
        Features are aligned by index with lines in the .txt file.
        Returns None if file doesn't exist.
    """
    features_path = features_dir / f"{image_id}.npz"
    if not features_path.exists():
        return None

    data = np.load(features_path)
    return data['features']  # (N, 257) float16


def _load_predictions_with_line_indices(
    preds_dir: Path,
    num_classes: int,
    confidence_threshold: float = 0.26,
) -> Dict[int, List[Tuple[str, int, np.ndarray, float]]]:
    """
    Load YOLO predictions from .txt files, preserving line order.
    
    This is different from _load_yolo_dataset which sorts by score.
    We need to preserve the original line order to align with features array.
    
    Note: Predictions are filtered by confidence_threshold BEFORE matching.
    This means the classifier and bbox regressor only see examples above this threshold.
    Consider using a lower threshold (e.g., 0.05) here to maximize training examples,
    then applying a higher operating threshold during evaluation.
    
    Args:
        preds_dir: Directory containing prediction .txt files
        num_classes: Number of classes
        confidence_threshold: Minimum score threshold
    
    Returns:
        Dict mapping class_id -> list of (img_id, line_idx, box_xyxy, score)
        where line_idx is the 0-based index of the line in the .txt file.
    """
    from src.eval_yolo import _yolo_to_xyxy  # Convert cxcywh -> xyxy
    
    preds_by_class: Dict[int, List[Tuple[str, int, np.ndarray, float]]] = {
        c: [] for c in range(num_classes)
    }
    
    # Get all prediction files
    pred_files = sorted(preds_dir.glob("*.txt"), key=_sort_key)
    
    for pred_file in pred_files:
        img_id = pred_file.stem
        
        with pred_file.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                
                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    continue
                
                # YOLO format: class cx cy w h [score]
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                score = float(parts[5]) if len(parts) >= 6 else 1.0
                
                if score < confidence_threshold:
                    continue
                
                # Convert to xyxy for IoU computation
                box_xyxy = _yolo_to_xyxy(cx, cy, w, h)
                
                # Store: (img_id, line_idx, box_xyxy, score)
                preds_by_class[class_id].append((img_id, line_idx, box_xyxy, score))
    
    return preds_by_class


def _xyxy_to_yolo(box: np.ndarray) -> np.ndarray:
    """
    Convert a box from xyxy normalized format to YOLO format (cx, cy, w, h).
    
    Args:
        box: Array of shape (4,) in xyxy format [x1, y1, x2, y2]
    
    Returns:
        Array of shape (4,) in YOLO format [cx, cy, w, h]
    """
    x1, y1, x2, y2 = box.astype(np.float32)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1)
    h = (y2 - y1)
    return np.asarray([cx, cy, w, h], dtype=np.float32)




def build_linear_probe_dataset_for_split(
    split: str,
    confidence_threshold: float = 0.05,
    iou_threshold: float = 0.5,
    features_dir: Optional[Path] = None,
) -> Path:
    """
    Build the linear-probe dataset for a single split (e.g. 'train').

    Args:
        split:
            Dataset split to process ('train', 'val', or 'test').
        confidence_threshold:
            Score threshold applied when loading SAM 3 predictions.
            Only predictions with score >= threshold are considered.
            Lower threshold (e.g., 0.05) maximizes training examples for the probe.
            The operating threshold can be higher during evaluation.
        iou_threshold:
            IoU threshold used to decide if a prediction is TP or FP,
            consistent with the detection evaluation (typically 0.5).
        features_dir:
            Optional path to directory containing .npz feature files.
            If None, defaults to data/processed/features/sam3_prehead/<split>/

    Returns:
        Path to the saved .npz file containing:
            - features: float32 array of shape (N, 257) where N=num_predictions
            - targets: int64 array of shape (N,) with values {0, 1}
            - class_ids: int64 array of shape (N,) with class indices
            - pred_boxes: float32 array of shape (N, 4) with predicted boxes in YOLO format (cx, cy, w, h)
            - gt_boxes: float32 array of shape (N, 4) with matched GT boxes in YOLO format (NaN for FPs)
    """
    # Get directories for ground-truth labels, SAM 3 predictions, and features.
    labels_dir = get_labels_dir(split)
    preds_dir = get_sam3_yolo_predictions_dir(split)
    
    if features_dir is None:
        features_dir = PROJECT_ROOT / "data" / "processed" / "features" / "sam3_prehead" / split
    
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features directory not found: {features_dir}. "
            f"Please run 'run_sam3_on_split.py' first to generate features."
        )
    
    num_classes = len(CLASS_PROMPTS)

    print(f"Building linear-probe dataset for split='{split}'")
    print(f"  Labels directory:      {labels_dir}")
    print(f"  Predictions directory: {preds_dir}")
    print(f"  Features directory:    {features_dir}")
    print(f"  Confidence threshold:  {confidence_threshold}")
    print(f"  IoU threshold:         {iou_threshold}")

    # Load GT using the standard function (only GT, not predictions)
    gt_by_class, _ = _load_yolo_dataset(
        labels_dir=labels_dir,
        preds_dir=preds_dir,  # Not used for GT loading
        num_classes=num_classes,
        confidence_threshold=0.0,  # Not relevant for GT
    )
    
    # Load predictions WITH line indices (instead of sorted by score)
    preds_by_class_with_idx = _load_predictions_with_line_indices(
        preds_dir=preds_dir,
        num_classes=num_classes,
        confidence_threshold=confidence_threshold,
    )

    # Build a mapping from image_id -> features for fast lookup
    print("  Loading feature files...")
    feature_cache: Dict[str, np.ndarray] = {}
    feature_files = list(features_dir.glob("*.npz"))
    for feat_file in feature_files:
        img_id = feat_file.stem
        feat_data = _load_features_for_image(features_dir, img_id)
        if feat_data is not None:
            feature_cache[img_id] = feat_data
    print(f"  Loaded features for {len(feature_cache)} images")

    all_features: List[np.ndarray] = []
    all_targets: List[int] = []
    all_class_ids: List[int] = []
    all_pred_boxes: List[np.ndarray] = []
    all_gt_boxes: List[np.ndarray] = []

    # Track images with missing features for final reporting
    missing_features_images: set[str] = set()

    # For per-class statistics / debugging: class_id -> (num_pos, num_neg)
    per_class_counts: Dict[int, Tuple[int, int]] = {}

    # Loop over all classes; we build TP/FP labels independently per class.
    for class_id in range(num_classes):
        gt_dict = gt_by_class.get(class_id, {})
        preds_with_idx = preds_by_class_with_idx.get(class_id, [])

        # Count how many GT boxes we have for this class (over all images).
        num_gt = sum(len(v) for v in gt_dict.values())
        if num_gt == 0 and not preds_with_idx:
            print(f"  Class {class_id}: no GT and no predictions, skipping.")
            continue

        print(
            f"  Class {class_id}: {num_gt} GT boxes, "
            f"{len(preds_with_idx)} predictions before matching."
        )

        # For each image, keep track of which GT boxes have already been
        # "claimed" by a higher-score prediction (greedy matching).
        matched: Dict[str, np.ndarray] = {
            img_id: np.zeros(len(boxes), dtype=bool)
            for img_id, boxes in gt_dict.items()
        }

        # Sort predictions by descending score for greedy TP/FP matching
        # Each prediction is: (img_id, line_idx, box_xyxy, score)
        preds_sorted = sorted(preds_with_idx, key=lambda x: x[3], reverse=True)

        num_pos = 0  # number of true positives for this class
        num_neg = 0  # number of false positives for this class

        # Iterate over predictions from highest to lowest score.
        for img_id, line_idx, box_pred, score in preds_sorted:
            gt_boxes = gt_dict.get(img_id)
            
            # Convert predicted box to YOLO format
            pred_box_yolo = _xyxy_to_yolo(box_pred)
            
            if gt_boxes is None or gt_boxes.size == 0:
                # No GT boxes for this image and class: prediction is a FP.
                target = 0
                best_idx = None
            else:
                # Compute IoU with all GT boxes in this image for this class.
                ious = _compute_iou(box_pred, gt_boxes)
                best_idx = int(np.argmax(ious))
                best_iou = float(ious[best_idx])

                if best_iou >= iou_threshold and not matched[img_id][best_idx]:
                    # This prediction becomes a TP and "claims" the GT box
                    # (so it cannot be matched again by lower-score predictions).
                    target = 1
                    matched[img_id][best_idx] = True
                else:
                    # Either IoU is too low, or the GT box was already claimed:
                    # in both cases, this prediction is treated as a FP.
                    target = 0
                    best_idx = None

            # Get feature directly using line_idx (perfect 1:1 alignment)
            # Features are aligned by index: features[i] corresponds to line i in .txt file
            img_features = feature_cache.get(img_id)
            if img_features is None:
                # Track this for reporting at end
                missing_features_images.add(img_id)
                continue
            
            if line_idx >= len(img_features):
                raise IndexError(
                    f"Image {img_id}: line_idx {line_idx} out of bounds (features has {len(img_features)} entries). "
                    f"This indicates .txt/.npz misalignment - check that files were generated together."
                )
            
            # Direct lookup: feature for this prediction is at features[line_idx]
            feats = img_features[line_idx].astype(np.float32)  # (257,)

            # Save GT box (if TP, take matched box; if FP, use NaN)
            if target == 1:
                gt_box_xyxy = gt_boxes[best_idx]  # (4,) xyxy normalizzato
                gt_box_yolo = _xyxy_to_yolo(gt_box_xyxy)
            else:
                gt_box_yolo = np.full((4,), np.nan, dtype=np.float32)

            # Append features and corresponding target (0/1) and class id.
            all_features.append(feats)
            all_targets.append(int(target))
            all_class_ids.append(int(class_id))
            all_pred_boxes.append(pred_box_yolo)
            all_gt_boxes.append(gt_box_yolo)
            
            # Update counters
            if target == 1:
                num_pos += 1
            else:
                num_neg += 1

        per_class_counts[class_id] = (num_pos, num_neg)
        print(
            f"    -> positives: {num_pos}, negatives: {num_neg} "
            f"(pos ratio: {num_pos / max(num_pos + num_neg, 1):.3f})"
        )

    # CRITICAL: Check if any images had missing features
    # Building a dataset with missing features would silently bias training
    if missing_features_images:
        raise FileNotFoundError(
            f"Cannot build linear probe dataset: {len(missing_features_images)} images have predictions but no features. "\
            f"Affected images (showing first 10): {sorted(list(missing_features_images))[:10]}. "\
            f"Please run 'run_sam3_on_split.py' for split='{split}' to generate features for all images."\
        )
    
    # If no predictions survived the confidence threshold, we cannot build a dataset.
    if not all_features:
        raise RuntimeError(
            f"No predictions found above threshold {confidence_threshold} "\
            f"for split '{split}'. Cannot build linear-probe dataset."\
        )

    # Stack and convert lists to final NumPy arrays.
    features_arr = np.stack(all_features, axis=0).astype(np.float32)
    targets_arr = np.asarray(all_targets, dtype=np.int64)
    class_ids_arr = np.asarray(all_class_ids, dtype=np.int64)
    pred_boxes_arr = np.stack(all_pred_boxes, axis=0).astype(np.float32)  # (N, 4)
    gt_boxes_arr = np.stack(all_gt_boxes, axis=0).astype(np.float32)  # (N, 4)

    # Create output directory for linear probe datasets if it does not exist.
    out_dir = PROJECT_ROOT / "data" / "processed" / "linear_probe"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sam3_linear_probe_{split}.npz"

    # Save compressed .npz with features, binary targets, class ids, and boxes.
    np.savez_compressed(
        out_path,
        features=features_arr,
        targets=targets_arr,
        class_ids=class_ids_arr,
        pred_boxes=pred_boxes_arr,
        gt_boxes=gt_boxes_arr,
    )

    print(f"\nSaved linear-probe dataset to: {out_path}")
    print(f"  Total samples: {features_arr.shape[0]}")
    print(f"  Feature dimension: {features_arr.shape[1]}")
    for class_id, (num_pos, num_neg) in per_class_counts.items():
        print(
            f"  Class {class_id}: {num_pos} pos, {num_neg} neg "
            f"(ratio={num_pos / max(num_pos + num_neg, 1):.3f})"
        )

    return out_path


def main() -> None:
    """
    Entry point for command-line use.

    By default, this builds the linear-probe dataset for the 'train' split.
    You can manually change the split to 'val' or 'test' if needed.
    """
    split = "train"
    confidence_threshold = 0.05  # Low threshold to maximize training examples
    iou_threshold = 0.5

    build_linear_probe_dataset_for_split(
        split=split,
        confidence_threshold=confidence_threshold,
        iou_threshold=iou_threshold,
    )


if __name__ == "__main__":
    main()
