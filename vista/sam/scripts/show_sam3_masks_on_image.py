"""
Visualization script: display SAM 3 binary segmentation masks overlaid on
the original image using class-specific color maps.

Redish tones = crashed car (class 0), Blue tones = person (class 1),
Green tones = undamaged car (class 2).
Edit the `split` and `image_id` variables inside `main()` to inspect different images.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import get_images_dir, get_sam3_segmentation_dir  # noqa: E402
from src.prompts import CLASS_PROMPTS  # noqa: E402


def load_masks_for_image(split: str, image_id: str) -> Dict[int, List[np.ndarray]]:
    """
    Load binary masks for a given image_id and split, grouped by class_id.
    """
    segm_dir = get_sam3_segmentation_dir(split)

    masks_by_class: Dict[int, List[np.ndarray]] = {}

    for class_id in CLASS_PROMPTS.keys():
        pattern = f"{image_id}_c{class_id}_i*.png"
        paths = sorted(segm_dir.glob(pattern))

        class_masks: List[np.ndarray] = []
        for p in paths:
            mask_img = Image.open(p).convert("L")
            mask_arr = (np.array(mask_img) > 0).astype(np.float32)
            class_masks.append(mask_arr)

        if class_masks:
            masks_by_class[class_id] = class_masks

    return masks_by_class


def main() -> None:
    split = "test"
    image_id = "118"  # change this to another id if needed

    images_dir = get_images_dir(split)
    img_path = images_dir / f"{image_id}.jpg"
    if not img_path.exists():
        img_path = images_dir / f"{image_id}.png"
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found for id={image_id} in {images_dir}")

    print(f"Using image: {img_path}")

    image = Image.open(img_path).convert("RGB")
    img_arr = np.array(image)

    masks_by_class = load_masks_for_image(split, image_id)

    if not masks_by_class:
        print("No masks found for this image. Make sure you ran run_sam3_on_split.py.")
        return

    fig, ax = plt.subplots()
    ax.imshow(img_arr)
    ax.axis("off")

    # Colors per class for visualization only
    class_cmaps = {
        0: "Reds",     # crashed car
        1: "Blues",    # person
        2: "Greens",   # undamaged car
    }

    for class_id, masks in masks_by_class.items():
        cmap = class_cmaps.get(class_id, "gray")
        for mask in masks:
            ax.imshow(
                np.ma.masked_where(mask == 0, mask),
                alpha=0.4,
                cmap=cmap,
            )

    plt.title(f"SAM 3 masks for image {image_id} (split={split})")
    plt.show()


if __name__ == "__main__":
    main()
