"""
Ultralytics-compatible wrapper around Sam3ImageModel.

``Sam3Model`` inherits from ``ultralytics.engine.model.Model`` so you get the
full ultralytics interface for free.  Inference is done by SAM 3 (text-prompt
segmentation); metrics are computed with the genuine ultralytics evaluation
stack (``DetectionValidator``, ``DetMetrics``).

Usage
-----
    from src.sam3_model import Sam3Model

    model = Sam3Model()                          # loads from HuggingFace
    model = Sam3Model("checkpoint.pt")           # loads from local file

    model.set_classes(
        names=["ambulance", "car"],
        prompts=["an ambulance vehicle", "a car"],  # optional richer prompts
    )

    # Validate — identical signature to YOLO().val()
    metrics = model.val(data="data/VistaSynth/data.yaml", split="val")
    print(metrics.box.map50)      # mAP@0.50
    print(metrics.box.map)        # mAP@0.50:0.95

    # Single-image inference
    results = model.predict("frame_0042.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils import callbacks as _ult_callbacks

from vista.models.validator import VISTAValidator

from .sam3_wrapper import Sam3ImageModel
from .yolo_export import YoloBox, nms_yolo_boxes, sam3_boxes_to_yolo


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────


def _yoloboxes_to_xyxy_px(boxes: List[YoloBox], w: int, h: int) -> torch.Tensor:
    """List[YoloBox] (normalized cx, cy, bw, bh) → float32 pixel xyxy tensor."""
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    coords = [
        [(b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h,
         (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h]
        for b in boxes
    ]
    return torch.tensor(coords, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Sam3Validator
# ─────────────────────────────────────────────────────────────────────────────


class Sam3Validator(VISTAValidator):
    """VISTAValidator for SAM3's file-based, per-class inference.

    Reuses all of DetectionValidator's metrics machinery via VISTAValidator.
    The inference step is replaced: SAM3 is called once per class per image
    using the original file paths from ``batch["im_file"]``, and the resulting
    boxes (in original image pixel space) are projected into the letterboxed
    pixel space that ``DetectionValidator._prepare_batch`` uses for GT boxes.
    """

    def _setup_device(self, model) -> torch.device:
        """Sam3ImageModel wraps the actual nn.Module at .model."""
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, "parameters"):
            return next(iter(inner.parameters())).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _infer(self, model, batch) -> list[dict[str, torch.Tensor]]:
        """Run SAM3 on each image in the batch.

        Args:
            model: ``Sam3ImageModel`` with ``.names`` (display names) and
                ``._prompts`` (text prompts dict) attached by ``Sam3Model.val()``.
            batch: Standard YOLO batch. ``batch["im_file"]`` carries the
                original file paths; ``batch["ori_shape"]`` and
                ``batch["ratio_pad"]`` carry the letterbox metadata.

        Returns:
            List of dicts – one per image – with keys ``bboxes`` (N, 4),
            ``conf`` (N,), ``cls`` (N,), ``extra`` (N, 0).  ``bboxes`` are
            in letterboxed pixel space to match the GT produced by
            ``DetectionValidator._prepare_batch``.
        """
        prompts: dict[int, str] = model._prompts
        conf_thr: float = self.args.conf
        iou_thr: float = self.args.iou
        max_det: int = self.args.max_det

        batch_preds: list[dict[str, torch.Tensor]] = []

        for si, im_file in enumerate(batch["im_file"]):
            ori_h, ori_w = batch["ori_shape"][si]   # original image (H, W)
            ratio_pad = batch["ratio_pad"][si]
            gain: float = ratio_pad[0][0]            # scale: original → letterboxed
            padw: float = ratio_pad[1][0]            # horizontal padding (px)
            padh: float = ratio_pad[1][1]            # vertical padding (px)

            # Run SAM3 once per class/prompt
            all_boxes: list[YoloBox] = []
            for cls_id, prompt in sorted(prompts.items()):
                pred = model.predict_with_text(im_file, prompt)
                all_boxes.extend(
                    sam3_boxes_to_yolo(
                        prediction=pred,
                        class_id=cls_id,
                        image_width=ori_w,
                        image_height=ori_h,
                        score_threshold=conf_thr,
                    )
                )
            all_boxes = nms_yolo_boxes(all_boxes, iou_threshold=iou_thr, max_det=max_det)

            n = len(all_boxes)
            if n > 0:
                # Boxes in original image pixel space
                boxes_orig = _yoloboxes_to_xyxy_px(all_boxes, ori_w, ori_h)
                # Project to letterboxed pixel space (matches DetectionValidator GT)
                boxes_lb = boxes_orig.clone()
                boxes_lb[:, [0, 2]] = boxes_lb[:, [0, 2]] * gain + padw
                boxes_lb[:, [1, 3]] = boxes_lb[:, [1, 3]] * gain + padh

                batch_preds.append({
                    "bboxes": boxes_lb.to(self.device),
                    "conf":   torch.tensor([b.score or 0.0 for b in all_boxes],
                                           dtype=torch.float32, device=self.device),
                    "cls":    torch.tensor([float(b.class_id) for b in all_boxes],
                                           dtype=torch.float32, device=self.device),
                    "extra":  torch.zeros((n, 0), dtype=torch.float32, device=self.device),
                })
            else:
                batch_preds.append({
                    "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                    "conf":   torch.zeros(0,       dtype=torch.float32, device=self.device),
                    "cls":    torch.zeros(0,       dtype=torch.float32, device=self.device),
                    "extra":  torch.zeros((0, 0),  dtype=torch.float32, device=self.device),
                })

        return batch_preds


# ─────────────────────────────────────────────────────────────────────────────
# Sam3Model
# ─────────────────────────────────────────────────────────────────────────────


class Sam3Model(Model):
    """
    SAM 3 wrapped as an ``ultralytics.engine.model.Model`` subclass.

    Inference is performed by SAM 3 (text-prompt segmentation).
    Metrics are computed with the genuine ultralytics evaluation stack so
    numbers are directly comparable to YOLO baselines.

    Parameters
    ----------
    checkpoint_path : str or None
        Path to a local SAM 3 checkpoint.  Pass *None* (default) to load
        from HuggingFace (requires internet on first run).
    """

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        # Bypass Model.__init__ (which expects a YOLO .pt file).
        torch.nn.Module.__init__(self)

        self.callbacks = _ult_callbacks.get_default_callbacks()
        self.predictor = None
        self.model = Sam3ImageModel(checkpoint_path)
        self.trainer = None
        self.ckpt = {}
        self.cfg = None
        self.ckpt_path = str(checkpoint_path) if checkpoint_path else None
        self.overrides = {"task": "detect"}
        self.metrics = None
        self.session = None
        self.task = "detect"
        self.model_name = "sam3"
        self._names: dict[int, str] = {}
        self._prompts: dict[int, str] = {}

    # ── task_map ─────────────────────────────────────────────────────────────

    @property
    def task_map(self) -> dict:
        return {"detect": {"validator": Sam3Validator}}

    # ── class vocabulary ──────────────────────────────────────────────────────

    @property
    def names(self) -> dict[int, str]:
        """Active class names as ``{class_id: name}``."""
        return self._names

    @names.setter
    def names(self, value: dict[int, str]) -> None:
        self._names = value

    def set_classes(
        self,
        names: list[str],
        prompts: Optional[list[str]] = None,
    ) -> None:
        """Set the detection vocabulary and optional text prompts.

        Args:
            names (list[str]): Display class names, e.g. ``["ambulance", "car"]``.
            prompts (list[str] | None): Richer text prompts passed to SAM3,
                e.g. ``["a white ambulance vehicle", "a passenger car"]``.
                Defaults to ``names`` when not provided.
        """
        self.names = {i: n for i, n in enumerate(names)}
        if prompts is not None:
            self._prompts = {i: p for i, p in enumerate(prompts)}
        else:
            self._prompts = dict(self.names)

    # ── predict ───────────────────────────────────────────────────────────────

    def predict(
        self,
        source: Union[str, Path, None] = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> list[Results]:
        """Run SAM3 on a single image and return standard Results objects.

        Args:
            source (str | Path): Path to the input image.
            prompts (dict[int, str] | None): ``{class_id: text_prompt}``.
                Falls back to ``self._prompts``, then to ``self.names``.
            conf (float): Minimum score to keep (default 0.0).
            iou (float): NMS IoU threshold (default 0.7).
            max_det (int): Max detections after NMS (default 300).

        Returns:
            list[Results]: Single-element list for API consistency.
        """
        if source is None:
            raise ValueError("'source' must be provided for Sam3Model.predict()")

        prompts: dict[int, str] = (
            kwargs.get("prompts")
            or self._prompts
            or self.names
        )
        conf: float = kwargs.get("conf", 0.0)
        iou: float = kwargs.get("iou", 0.7)
        max_det: int = kwargs.get("max_det", 300)

        img_path = Path(source)
        image = Image.open(img_path).convert("RGB")
        orig_np = np.array(image)
        w, h = image.size  # PIL: (width, height)

        all_boxes: list[YoloBox] = []
        for cls_id, prompt in sorted(prompts.items()):
            pred = self.model.predict_with_text(img_path, prompt)
            all_boxes.extend(
                sam3_boxes_to_yolo(
                    prediction=pred,
                    class_id=cls_id,
                    image_width=w,
                    image_height=h,
                    score_threshold=conf,
                )
            )
        all_boxes = nms_yolo_boxes(all_boxes, iou_threshold=iou, max_det=max_det)

        names = self.names or {i: p for i, p in prompts.items()}

        if all_boxes:
            boxes_xyxy = _yoloboxes_to_xyxy_px(all_boxes, w, h)
            confs = torch.tensor([b.score or 0.0 for b in all_boxes], dtype=torch.float32)
            clses = torch.tensor([float(b.class_id) for b in all_boxes], dtype=torch.float32)
            boxes_t = torch.cat([boxes_xyxy, confs.unsqueeze(1), clses.unsqueeze(1)], dim=1)
        else:
            boxes_t = torch.zeros((0, 6), dtype=torch.float32)

        return [Results(orig_np, path=str(img_path), names=names, boxes=boxes_t)]

    def __call__(self, source=None, stream=False, **kwargs):
        return self.predict(source, stream, **kwargs)

    # ── val ───────────────────────────────────────────────────────────────────

    def val(self, **kwargs) -> Any:
        """Validate using a YOLO-format dataset.

        Delegates to ``Model.val()`` which instantiates ``Sam3Validator``
        from the task map.  Pass ``data="path/to/dataset.yaml"`` and any
        other standard ultralytics val kwargs.

        Returns:
            DetMetrics: Standard Ultralytics detection metrics.
        """
        if not self._prompts and not self._names:
            from .prompts import CLASS_PROMPTS
            self.set_classes(
                names=list(CLASS_PROMPTS.values()),
                prompts=list(CLASS_PROMPTS.values()),
            )

        # Sam3Validator reads these from the model object passed to it.
        self.model.names = self.names
        self.model._prompts = self._prompts or self.names

        return super().val(**kwargs)
