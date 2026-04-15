"""
Utilities to export SAM 3 segmentation masks to image files.

This module takes the raw masks + scores predicted by SAM 3 for a single
image and converts them to NumPy arrays and PNG files. It is used by
the scripts that run inference over a dataset split to save one binary
mask per prediction (optionally filtered by a score threshold).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from .sam3_wrapper import Sam3Prediction


def _prediction_masks_to_numpy(prediction: Sam3Prediction) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert SAM 3 masks and scores to numpy arrays.

    Masks are returned as an array of shape (N, H, W) with float values in [0, 1].
    """
    # Extract raw mask and score tensors from the SAM 3 prediction.
    masks = prediction.masks
    scores = prediction.scores

    # If masks/scores are tensors (e.g. PyTorch), move them to CPU and convert to NumPy.
    if hasattr(masks, "detach"):
        masks_np = masks.detach().cpu().numpy()
    else:
        masks_np = np.asarray(masks)

    # SAM 3 masks are often shaped as (N, 1, H, W): drop the channel dimension.
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        # (N, 1, H, W) -> (N, H, W)
        masks_np = masks_np[:, 0, :, :]
    elif masks_np.ndim != 3:
        # If it is neither (N,1,H,W) nor (N,H,W), something is probably wrong.
        raise ValueError(f"Unexpected mask shape: {masks_np.shape}")

    if hasattr(scores, "detach"):
        scores_np = scores.detach().cpu().numpy()
    else:
        scores_np = np.asarray(scores)

    return masks_np, scores_np


def save_sam3_masks_for_image(
    prediction: Sam3Prediction,
    class_id: int,
    image_id: str,
    output_root: Path,
    score_threshold: float = 0.0,
    max_masks: Optional[int] = None,
) -> List[Path]:
    """
    Save binary segmentation masks for a single image and class.

    Masks are saved as 8-bit PNG images in:
        output_root / f"{image_id}_c{class_id}_i{idx}.png"
    """
    # Convert SAM 3 masks/scores to NumPy arrays ready for processing.
    masks_np, scores_np = _prediction_masks_to_numpy(prediction)

    # Ensure the output directory exists (creates parent dirs if needed).
    output_root.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    count = 0  # number of masks actually saved

    # Iterate over all masks produced by SAM 3 for this image/class.
    for idx, (mask, score) in enumerate(zip(masks_np, scores_np)):
        # Filter by confidence threshold.
        if score < score_threshold:
            continue

        # If a maximum number of masks is set, respect that limit.
        if max_masks is not None and count >= max_masks:
            break

        # Threshold mask to binary and convert to uint8 image.
        # mask is float in [0,1]; here it is thresholded to {0,255}
        # and converted to an 8-bit image.
        mask_bin = (mask > 0.5).astype(np.uint8) * 255
        mask_img = Image.fromarray(mask_bin)

        # Build a file name encoding image id, class id and mask index.
        out_name = f"{image_id}_c{class_id}_i{idx}.png"
        out_path = output_root / out_name
        mask_img.save(out_path)

        saved_paths.append(out_path)
        count += 1

    return saved_paths
