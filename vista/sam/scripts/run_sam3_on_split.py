"""
Run SAM 3 over a given dataset split and export predictions.

For each image in the chosen split, this script:
- queries SAM 3 once per class using the corresponding text prompt,
- converts the resulting detections to YOLO-style bounding boxes,
- applies class-wise NMS to reduce duplicate boxes,
- writes predictions to .txt files (one per image),
- and optionally saves binary segmentation masks as PNGs.

Usage:
  # Run on test split with default settings
  python run_sam3_on_split.py
  
  # Run on train split
  python run_sam3_on_split.py --split train
  
  # Custom export threshold (low to keep candidates)
  python run_sam3_on_split.py --split val --export_threshold 0.10
  
  # Limit to first 10 images for testing
  python run_sam3_on_split.py --split test --max_images 10
  
  # Save segmentation masks (useful for qualitative analysis)
  python run_sam3_on_split.py --split test --save_segmentations
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from time import time
from typing import Optional, List

import numpy as np
from PIL import Image

# Add project root to PYTHONPATH so that "src" can be imported when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    get_images_dir,
    get_sam3_yolo_predictions_dir,
    get_sam3_segmentation_dir,
    EXPORT_THRESHOLD_DEFAULT,
    NMS_IOU_DEFAULT,
    NMS_MAX_DET_DEFAULT,
)
from src.prompts import CLASS_PROMPTS  # noqa: E402
from src.sam3_wrapper import Sam3ImageModel  # noqa: E402
from src.yolo_export import (  # noqa: E402
    sam3_boxes_to_yolo,
    yolo_boxes_to_lines,
    YoloBox,
    nms_yolo_boxes,
)
from src.segmentation_export import save_sam3_masks_for_image  # noqa: E402


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


def run_sam3_on_split(
    split: str,
    export_threshold: float = EXPORT_THRESHOLD_DEFAULT,
    max_images: Optional[int] = None,
    save_segmentations: bool = False,
    max_masks_per_image_per_class: Optional[int] = None,
    nms_iou: float = NMS_IOU_DEFAULT,
    nms_max_det: int = NMS_MAX_DET_DEFAULT,
) -> None:
    """
    Run SAM 3 on all images of a given split and save YOLO-style predictions
    and, optionally, segmentation masks.

    Args:
        split: Dataset split to process ("train", "val", or "test").
        export_threshold: Minimum confidence for detections/masks to be EXPORTED.
                         Should be LOW (e.g., 0.05-0.10) to keep candidate detections
                         that the linear probe can later refine. The evaluation threshold
                         (used during metric computation) is separate and chosen via sweep.
                         Default: EXPORT_THRESHOLD_DEFAULT (0.05)
        max_images: Optional limit on the number of images to process.
        save_segmentations: If True, also export binary segmentation masks as PNG.
                           Default: False (masks not needed for linear probe experiments).
        max_masks_per_image_per_class: Optional max number of masks saved per image/class.
        nms_iou: IoU threshold used for class-wise NMS.
        nms_max_det: Global cap on the number of boxes kept after NMS per image.
    """
    # Resolve the directory containing the input images for this split.
    images_dir = get_images_dir(split)

    # Resolve and create (if needed) the directory where YOLO predictions will be written.
    pred_dir = get_sam3_yolo_predictions_dir(split)
    pred_dir.mkdir(parents=True, exist_ok=True)

    segm_dir = None
    if save_segmentations:
        # Resolve and create (if needed) the directory where segmentation masks will be saved.
        segm_dir = get_sam3_segmentation_dir(split)
        segm_dir.mkdir(parents=True, exist_ok=True)

    # Create directory for feature embeddings (.npz files)
    features_dir = PROJECT_ROOT / "data" / "processed" / "features" / "sam3_prehead" / split
    features_dir.mkdir(parents=True, exist_ok=True)

    # Collect all JPG and PNG images, sorted safely (numeric if possible, lexicographic otherwise).
    image_files = sorted(
        list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")),
        key=_sort_key,
    )
    if not image_files:
        # If there are no images, fail fast with a clear error.
        raise RuntimeError(f"No images found in {images_dir}")

    # Optionally restrict to the first max_images images for quicker experiments.
    if max_images is not None:
        image_files = image_files[:max_images]

    print(f"Split: {split}")
    print(f"Number of images: {len(image_files)}")
    print(f"Images directory: {images_dir}")
    print(f"Prediction directory: {pred_dir}")
    print(f"Export threshold (for saving candidates): {export_threshold}")
    print(f"NMS: iou={nms_iou}, max_det={nms_max_det}")
    print(f"Features directory: {features_dir}")
    if save_segmentations:
        print(f"Segmentation directory: {segm_dir}")

    # Instantiate a single SAM 3 model to be reused for all images/classes.
    model = Sam3ImageModel()
    t_start = time()

    # Main loop over all images in the split.
    for idx, img_path in enumerate(image_files, start=1):
        # Load the image in RGB mode.
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        # Accumulate YOLO-format boxes from all classes for this image.
        all_boxes: List[YoloBox] = []
        image_id = img_path.stem

        # Loop over all classes defined in CLASS_PROMPTS and query SAM 3 per class.
        for class_id, prompt in CLASS_PROMPTS.items():
            # Run SAM 3 for the current image and text prompt (single class).
            prediction = model.predict_with_text(img_path, prompt)

            # Convert SAM 3 detections (pixel boxes) into YOLO-normalized boxes.
            yolo_boxes = sam3_boxes_to_yolo(
                prediction=prediction,
                class_id=class_id,
                image_width=width,
                image_height=height,
                score_threshold=export_threshold,
            )
            all_boxes.extend(yolo_boxes)

            # Optionally also export segmentation masks for this class.
            if save_segmentations and segm_dir is not None:
                save_sam3_masks_for_image(
                    prediction=prediction,
                    class_id=class_id,
                    image_id=image_id,
                    output_root=segm_dir,
                    score_threshold=export_threshold,
                    max_masks=max_masks_per_image_per_class,
                )

        # Apply class-wise NMS to all boxes collected for this image
        # (this will remove duplicates per class).
        before = len(all_boxes)
        all_boxes = nms_yolo_boxes(all_boxes, iou_threshold=nms_iou, max_det=nms_max_det)
        after = len(all_boxes)

        # Convert YoloBox instances to YOLO-style text lines, including score as 6th column.
        lines = yolo_boxes_to_lines(
            all_boxes,
            include_score_column=True,
            include_score_comment=False,
        )

        # Write predictions for this image to a .txt file with the same stem as the image.
        out_path = pred_dir / f"{image_id}.txt"
        
        # CRITICAL: Check that ALL boxes have features before writing anything
        # This ensures perfect alignment between .txt lines and .npz features
        boxes_without_features = [i for i, box in enumerate(all_boxes) if box.features is None]
        if boxes_without_features:
            raise RuntimeError(
                f"Image {image_id}: {len(boxes_without_features)} boxes missing features "
                f"(indices: {boxes_without_features[:5]}...). Cannot guarantee .txt/.npz alignment. "
                f"This should never happen - check SAM3 feature extraction."
            )
        
        # Now safe to write: all boxes have features
        with out_path.open("w", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")

        # Save features aligned with the final boxes (after NMS)
        # CRITICAL: Features must be in same order as lines in .txt file
        # We concatenate [query_embedding_256d, score] to get 257-d features
        box_features = []
        
        for box in all_boxes:
            # We already verified all boxes have features above
            # Concatenate 256-d query embedding with original score (1-d) -> 257-d total
            score_val = box.score if box.score is not None else 0.0
            # FIX: np.concatenate doesn't accept dtype parameter
            feat_with_score = np.concatenate([box.features, [score_val]]).astype(np.float32)
            box_features.append(feat_with_score)

        # Save to .npz with features aligned to .txt lines
        # IMPORTANT: Always save .npz even if empty (0 detections) to maintain consistency
        # This prevents FileNotFoundError in apply_linear_probe for images with no detections
        if box_features:
            features_arr = np.array(box_features, dtype=np.float16)  # (N, 257)
        else:
            # Empty array with correct shape: (0, 257)
            features_arr = np.empty((0, 257), dtype=np.float16)

        features_path = features_dir / f"{image_id}.npz"
        np.savez_compressed(
            features_path,
            features=features_arr,  # (N, 257): [256-d query + 1-d score]
        )

        elapsed = time() - t_start
        print(
            f"[{idx}/{len(image_files)}] {img_path.name} -> {out_path.name} "
            f"({before}->{after} boxes after NMS, {len(box_features) if box_features else 0} features, elapsed {elapsed:.1f}s)"
        )

    total_time = time() - t_start
    print(f"Done. Processed {len(image_files)} images in {total_time:.1f}s.")


def main() -> None:
    """
    CLI entry point for running SAM 3 over a chosen split.
    
    Supports command-line arguments for all major parameters to make the
    script reproducible and scriptable without modifying code.
    """
    parser = argparse.ArgumentParser(
        description="Run SAM 3 on a dataset split and export predictions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run on test split with defaults
  python run_sam3_on_split.py
  
  # Run on train split
  python run_sam3_on_split.py --split train
  
  # Custom export threshold (keep more candidates)
  python run_sam3_on_split.py --split val --export_threshold 0.10
  
  # Limit to first 10 images for testing
  python run_sam3_on_split.py --split test --max_images 10
  
  # Save segmentation masks (for qualitative analysis)
  python run_sam3_on_split.py --split test --save_segmentations
  
  # Custom NMS parameters
  python run_sam3_on_split.py --split val --nms_iou 0.6 --nms_max_det 500
        """
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to process (default: test).",
    )
    parser.add_argument(
        "--export_threshold",
        type=float,
        default=EXPORT_THRESHOLD_DEFAULT,
        help=f"Export threshold for candidate detections (default: {EXPORT_THRESHOLD_DEFAULT}). "
             "Should be LOW to keep candidates for linear probe.",
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=None,
        help="Maximum number of images to process (default: all).",
    )
    parser.add_argument(
        "--save_segmentations",
        action="store_true",
        help="Save binary segmentation masks as PNG (default: False).",
    )
    parser.add_argument(
        "--max_masks_per_image_per_class",
        type=int,
        default=None,
        help="Maximum number of masks to save per image per class (default: all).",
    )
    parser.add_argument(
        "--nms_iou",
        type=float,
        default=NMS_IOU_DEFAULT,
        help=f"IoU threshold for NMS (default: {NMS_IOU_DEFAULT}).",
    )
    parser.add_argument(
        "--nms_max_det",
        type=int,
        default=NMS_MAX_DET_DEFAULT,
        help=f"Maximum detections per image after NMS (default: {NMS_MAX_DET_DEFAULT}).",
    )
    
    args = parser.parse_args()

    run_sam3_on_split(
        split=args.split,
        export_threshold=args.export_threshold,
        max_images=args.max_images,
        save_segmentations=args.save_segmentations,
        max_masks_per_image_per_class=args.max_masks_per_image_per_class,
        nms_iou=args.nms_iou,
        nms_max_det=args.nms_max_det,
    )


if __name__ == "__main__":
    main()
