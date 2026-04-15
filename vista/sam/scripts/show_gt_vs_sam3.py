"""
Visualization script: overlay ground-truth and SAM 3 predicted bounding boxes
side-by-side on a single image to allow qualitative inspection of detections.

Ground-truth boxes are drawn in green; predicted boxes in red.
Edit the `split` and `image_id` variables inside `main()` to inspect different images.
"""

import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
from PIL import Image

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_images_dir,
    get_labels_dir,
    get_sam3_yolo_predictions_dir,
)


def load_yolo_boxes(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Load YOLO-format boxes from a text file: class_id cx cy w h.
    """
    boxes: List[Tuple[int, float, float, float, float]] = []

    if not label_path.exists():
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            boxes.append((class_id, cx, cy, w, h))

    return boxes


def yolo_to_xyxy(
    boxes: List[Tuple[int, float, float, float, float]],
    img_width: int,
    img_height: int,
) -> List[Tuple[int, float, float, float, float]]:
    """
    Convert YOLO boxes (normalized) to pixel coordinates [x1, y1, x2, y2].
    """
    result: List[Tuple[int, float, float, float, float]] = []

    for class_id, cx, cy, w, h in boxes:
        cx_px = cx * img_width
        cy_px = cy * img_height
        w_px = w * img_width
        h_px = h * img_height

        x1 = cx_px - w_px / 2.0
        y1 = cy_px - h_px / 2.0
        x2 = cx_px + w_px / 2.0
        y2 = cy_px + h_px / 2.0

        result.append((class_id, x1, y1, x2, y2))

    return result


def main() -> None:
    split = "val"
    image_id = "1"  # change to "10", "11", "12", "13" to inspect other images

    images_dir = get_images_dir(split)
    labels_dir = get_labels_dir(split)
    preds_dir = get_sam3_yolo_predictions_dir(split)

    img_path = images_dir / f"{image_id}.jpg"
    if not img_path.exists():
        img_path = images_dir / f"{image_id}.png"

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found for id={image_id} in {images_dir}")

    gt_path = labels_dir / f"{image_id}.txt"
    pred_path = preds_dir / f"{image_id}.txt"

    print(f"Image: {img_path}")
    print(f"GT labels: {gt_path}")
    print(f"SAM3 predictions: {pred_path}")

    image = Image.open(img_path).convert("RGB")
    width, height = image.size

    gt_boxes_yolo = load_yolo_boxes(gt_path)
    pred_boxes_yolo = load_yolo_boxes(pred_path)

    gt_boxes = yolo_to_xyxy(gt_boxes_yolo, width, height)
    pred_boxes = yolo_to_xyxy(pred_boxes_yolo, width, height)

    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"Predicted boxes: {len(pred_boxes)}")

    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis("off")

    # Ground truth boxes in green
    for class_id, x1, y1, x2, y2 in gt_boxes:
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=2,
            edgecolor="green",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            max(y1 - 5, 0),
            f"GT {class_id}",
            fontsize=7,
            color="green",
            backgroundcolor="white",
        )

    # Predicted boxes in red
    for class_id, x1, y1, x2, y2 in pred_boxes:
        rect = plt.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            fill=False,
            linewidth=1.5,
            edgecolor="red",
        )
        ax.add_patch(rect)
        ax.text(
            x1,
            y2 + 5,
            f"P {class_id}",
            fontsize=7,
            color="red",
            backgroundcolor="white",
        )

    plt.show()


if __name__ == "__main__":
    main()
