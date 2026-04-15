from __future__ import annotations

"""
Apply the trained linear probe to SAM 3 YOLO-style detections.

This script:
- Loads the original SAM 3 predictions for a given split (e.g. "test"),
  stored in YOLO format in:

      data/processed/predictions/sam3_yolo/<split>/*.txt

  Each line is expected to be:
      class_id  cx  cy  w  h  score

- Loads semantic features (257-d: 256-d query embeddings + score) from .npz files,
  aligned by line index with the .txt file.

- Applies the class-specific logistic regression weights learned by
  `train_linear_probe.py`, stored in:

      data/processed/linear_probe/sam3_linear_probe_weights.npz

- Writes new YOLO files, with UPDATED boxes (if bbox refinement is enabled) and
  UPDATED scores, to:

      data/processed/predictions/sam3_linear_probe_yolo/<split>/*.txt

- If bbox refinement weights are available, also applies bbox delta corrections
  and re-runs NMS on the refined boxes.

The updated scores can then be evaluated with `eval_yolo.py` in the same
way as the original SAM 3 predictions.
"""

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_sam3_yolo_predictions_dir,
    PREDICTIONS_DIR,
    NMS_IOU_DEFAULT,
    NMS_MAX_DET_DEFAULT,
)
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.yolo_export import YoloBox, nms_yolo_boxes  # noqa: E402


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





def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x_clipped = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))


def apply_linear_probe_to_split(
    split: str = "test",
    features_dir: Optional[Path] = None,
) -> Path:
    """
    Re-score SAM 3 predictions on a given split using the trained linear probe.

    Args:
        split:
            Dataset split to process ('train', 'val', or 'test').
        features_dir:
            Optional path to directory containing .npz feature files.
            If None, defaults to data/processed/features/sam3_prehead/<split>/

    Returns:
        Path to the directory containing the re-scored YOLO prediction files.
    """
    # Where the original SAM 3 predictions are stored
    in_dir = get_sam3_yolo_predictions_dir(split)

    # Where we will write the re-scored predictions
    out_dir = PREDICTIONS_DIR / "sam3_linear_probe_yolo" / split
    out_dir.mkdir(parents=True, exist_ok=True)

    # Where the features are stored
    if features_dir is None:
        features_dir = PROJECT_ROOT / "data" / "processed" / "features" / "sam3_prehead" / split
    
    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features directory not found: {features_dir}. "
            f"Please run 'run_sam3_on_split.py' first to generate features."
        )

    # Load class-wise weights and biases learned by train_linear_probe.py
    weights_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "linear_probe"
        / "sam3_linear_probe_weights.npz"
    )
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Linear probe weights not found: {weights_path}. "
            "Please run 'train_linear_probe.py' first."
        )

    weights_data = np.load(weights_path)
    all_weights = weights_data["weights"]  # shape (num_classes, D)
    all_biases = weights_data["biases"]    # shape (num_classes,)
    
    # Check if bbox regressor weights are present
    has_bbox = ("bbox_weights" in weights_data.files) and ("bbox_biases" in weights_data.files)
    if has_bbox:
        bbox_W = weights_data["bbox_weights"]  # (C, D+4, 4)
        bbox_b = weights_data["bbox_biases"]   # (C, 4)
        print("  BBox refinement: ENABLED (weights found)")
    else:
        bbox_W = None
        bbox_b = None
        print("  BBox refinement: DISABLED (weights not found)")

    num_classes = len(CLASS_PROMPTS)
    if all_weights.shape[0] != num_classes:
        raise ValueError(
            f"Mismatch between number of classes in CLASS_PROMPTS ({num_classes}) "
            f"and weights shape ({all_weights.shape[0]})."
        )

    feature_dim = all_weights.shape[1]
    print(f"Applying linear probe on split='{split}'")
    print(f"  Input predictions dir:  {in_dir}")
    print(f"  Output predictions dir: {out_dir}")
    print(f"  Features directory:     {features_dir}")
    print(f"  Num classes:            {num_classes}")
    print(f"  Feature dimension:      {feature_dim}")

    # Collect all YOLO prediction files, sorted safely (numeric when possible).
    pred_files: List[Path] = sorted(
        in_dir.glob("*.txt"),
        key=_sort_key,
    )
    if not pred_files:
        raise RuntimeError(f"No prediction files found in {in_dir}")

    # Loop over prediction files (one per image).
    for idx, pred_path in enumerate(pred_files, start=1):
        image_id = pred_path.stem
        out_path = out_dir / f"{image_id}.txt"

        # Load features for this image
        feat_data = _load_features_for_image(features_dir, image_id)
        if feat_data is None:
            # CRITICAL: If we have predictions (.txt exists) but no features (.npz missing),
            # this is a data integrity issue - fail explicitly rather than skip silently
            raise FileNotFoundError(
                f"Missing features for image {image_id}. "
                f"Predictions file exists at {pred_path} but no corresponding .npz found in {features_dir}. "
                f"Please re-run 'run_sam3_on_split.py' to regenerate features."
            )
        
        # Validate feature dimensions match expected (257-d: 256-d query + score)
        if len(feat_data.shape) != 2 or feat_data.shape[1] != feature_dim:
            raise ValueError(
                f"Image {image_id}: feature dimension mismatch. "
                f"Expected shape (N, {feature_dim}), got {feat_data.shape}. "
                f"This may indicate old features or pipeline version mismatch."
            )

        new_lines: List[str] = []

        # Read original SAM 3 predictions for this image
        # Use enumerate to track line index for feature alignment
        line_idx = -1
        with pred_path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                parts = line.strip().split()
                # Expect at least class + 4 coords; if malformed, this is a corruption issue
                if len(parts) < 5:
                    raise ValueError(
                        f"Malformed line {line_idx} in {pred_path}: expected at least 5 fields, got {len(parts)}. "
                        f"Line content: '{line.strip()}'. This may indicate file corruption."
                    )

                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    # CRITICAL: Cannot skip this line - would break feature alignment
                    # If we skip with continue, line_idx advances but we don't consume feature
                    raise ValueError(
                        f"Invalid class_id {class_id} in {pred_path} line {line_idx}. "
                        f"Expected 0 <= class_id < {num_classes}. This may indicate data corruption."
                    )

                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                # If no score column is present, default to 1.0
                score_original = float(parts[5]) if len(parts) >= 6 else 1.0

                # Get feature directly using line_idx (perfect 1:1 alignment)
                if line_idx >= len(feat_data):
                    # This indicates .txt/.npz misalignment - fail explicitly
                    raise IndexError(
                        f"Image {image_id}: line_idx {line_idx} out of bounds (features has {len(feat_data)} entries). "
                        f"This indicates .txt/.npz misalignment - files may not have been generated together. "
                        f"Please re-run 'run_sam3_on_split.py' for split='{split}'."
                    )
                    
                # Direct lookup: feature for this prediction is at features[line_idx]
                feats = feat_data[line_idx].astype(np.float32)  # (257,)
                
                # Apply bbox refinement if available
                if has_bbox:
                    # Concatenate features + predicted box
                    x_reg = np.concatenate(
                        [feats, np.asarray([cx, cy, w, h], dtype=np.float32)],
                        axis=0
                    )  # (D+4,)
                    
                    # Predict deltas: [dx, dy, dw, dh]
                    delta = x_reg @ bbox_W[class_id] + bbox_b[class_id]  # (4,)
                    
                    # Clamp deltas to reduce outliers (same as training)
                    delta = np.clip(delta, [-0.5, -0.5, -2.0, -2.0], [0.5, 0.5, 2.0, 2.0])
                    dx, dy, dw, dh = map(float, delta)
                    
                    # Apply deltas to refine box coordinates
                    # Avoid division/exp instabilities
                    w_safe = max(w, 1e-6)
                    h_safe = max(h, 1e-6)
                    
                    cx = cx + dx * w_safe
                    cy = cy + dy * h_safe
                    w = w_safe * float(np.exp(np.clip(dw, -4.0, 4.0)))
                    h = h_safe * float(np.exp(np.clip(dh, -4.0, 4.0)))
                    
                    # Clamp to valid ranges
                    cx = float(np.clip(cx, 0.0, 1.0))
                    cy = float(np.clip(cy, 0.0, 1.0))
                    w = float(np.clip(w, 1e-6, 1.0))
                    h = float(np.clip(h, 1e-6, 1.0))
                
                # Use the linear probe to compute new score
                w_c = all_weights[class_id]  # weights for this class (D,)
                b_c = float(all_biases[class_id])

                logit = float(feats @ w_c + b_c)
                score_new = float(_sigmoid(np.asarray(logit)))

                # Ensure score is in [0, 1]
                score_new = float(np.clip(score_new, 0.0, 1.0))

                # Store refined box and score
                new_lines.append((class_id, cx, cy, w, h, score_new))
        
        # CRITICAL: Verify that we processed exactly as many lines as we have features
        # This catches cases where .txt and .npz are misaligned (extra features or missing lines)
        num_lines_processed = line_idx + 1 if 'line_idx' in locals() else 0
        if num_lines_processed != len(feat_data):
            raise ValueError(
                f"Image {image_id}: .txt/.npz alignment error. "
                f"Processed {num_lines_processed} lines but have {len(feat_data)} features. "
                f"Files may not have been generated together. Please re-run 'run_sam3_on_split.py'."
            )
        
        # Apply NMS if bbox refinement was performed (boxes changed)
        if has_bbox and new_lines:
            # Convert to YoloBox objects
            boxes = [
                YoloBox(
                    class_id=int(item[0]),
                    cx=float(item[1]),
                    cy=float(item[2]),
                    w=float(item[3]),
                    h=float(item[4]),
                    score=float(item[5]),
                )
                for item in new_lines
            ]
            
            # Apply NMS
            boxes = nms_yolo_boxes(
                boxes,
                iou_threshold=NMS_IOU_DEFAULT,
                max_det=NMS_MAX_DET_DEFAULT,
            )
            
            # Convert back to text format
            final_lines = [
                f"{box.class_id} {box.cx:.6f} {box.cy:.6f} {box.w:.6f} {box.h:.6f} {box.score:.6f}"
                for box in boxes
            ]
        else:
            # No bbox refinement: convert tuples to formatted strings
            final_lines = [
                f"{item[0]} {item[1]:.6f} {item[2]:.6f} {item[3]:.6f} {item[4]:.6f} {item[5]:.6f}"
                for item in new_lines
            ]

        # Write updated predictions for this image
        with out_path.open("w", encoding="utf-8") as f_out:
            for ln in final_lines:
                f_out.write(ln + "\n")

        if idx % 50 == 0 or idx == len(pred_files):
            print(f"  [{idx}/{len(pred_files)}] Processed {pred_path.name} -> {out_path.name}")

    print("Done applying linear probe.")
    return out_dir


def main() -> None:
    """
    Entry point for command-line use.

    By default, this applies the linear probe to the 'test' split.
    """
    split = "test"
    apply_linear_probe_to_split(split=split)


if __name__ == "__main__":
    main()
