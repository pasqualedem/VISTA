from pathlib import Path

# Project root directory (contains README.md, data/, src/, etc.)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

# Dataset-specific paths
DATASET_YAML_PATH = RAW_DATA_DIR / "data.yaml"
IMAGES_ROOT = RAW_DATA_DIR / "images"
LABELS_ROOT = RAW_DATA_DIR / "labels"


def _check_split(split: str) -> str:
    """Validate that the split name is one of the expected values."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split: {split!r} (expected 'train', 'val', or 'test')")
    return split


def get_images_dir(split: str):
    """Return the images directory for the given split."""
    split = _check_split(split)
    return IMAGES_ROOT / split


def get_labels_dir(split: str):
    """Return the labels directory for the given split."""
    split = _check_split(split)
    return LABELS_ROOT / split

# Directory for model predictions (not tracked by git, under data/processed).
PREDICTIONS_DIR = DATA_DIR / "processed" / "predictions"


def get_sam3_yolo_predictions_dir(split: str) -> Path:
    """
    Return the directory where YOLO-style predictions produced by SAM 3
    will be stored for a given split (train, val, test).
    """
    split = _check_split(split)
    return PREDICTIONS_DIR / "sam3_yolo" / split

# Directory for segmentation outputs.
SEGMENTATIONS_DIR = DATA_DIR / "processed" / "segmentations"


def get_sam3_segmentation_dir(split: str) -> Path:
    """
    Return the directory where SAM 3 segmentation masks will be stored
    for a given split (train, val, test).
    """
    split = _check_split(split)
    return SEGMENTATIONS_DIR / "sam3" / split


# ==============================================================================
# Default thresholds and hyperparameters (for reproducibility and consistency)
# ==============================================================================

# Export threshold: used during prediction export (run_sam3_on_split.py).
# Should be LOW to keep candidate detections for the linear probe to refine.
# The linear probe can adjust confidence scores, but only for boxes that were saved.
EXPORT_THRESHOLD_DEFAULT = 0.05

# Evaluation threshold: used during metric computation (eval_sam3_on_split.py).
# Should be chosen via threshold sweep on the validation set (see --sweep mode).
# This is the "operating point" for production inference.
EVAL_THRESHOLD_DEFAULT = 0.26  # Can be updated after validation sweep

# NMS (Non-Maximum Suppression) parameters
NMS_IOU_DEFAULT = 0.7       # IoU threshold for NMS (higher = more boxes kept)
NMS_MAX_DET_DEFAULT = 300   # Maximum detections per image after NMS

# Note: Matching IoU threshold for TP/FP assignment during evaluation is fixed at 0.5
# in src/eval_yolo.py (evaluate_yolo_predictions function). This is standard for mAP@50.
