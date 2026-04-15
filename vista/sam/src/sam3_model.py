"""
Ultralytics-compatible wrapper around Sam3ImageModel.

``Sam3Model`` inherits from ``ultralytics.engine.model.Model`` so you get the
full ultralytics interface for free.  Inference is done by SAM 3 (text-prompt
segmentation); metrics are computed with the genuine ultralytics evaluation
stack (``DetMetrics``, ``match_predictions``, ``box_iou``).

Usage
-----
    from src.sam3_model import Sam3Model

    model = Sam3Model()                          # loads from HuggingFace
    model = Sam3Model("checkpoint.pt")           # loads from local file

    # Validate — identical signature to YOLO().val()
    metrics = model.val(data="data/VistaSynth/data.yaml", split="val")
    print(metrics.box.map50)      # mAP@0.50
    print(metrics.box.map)        # mAP@0.50:0.95

    # Single-image inference
    results = model.predict("frame_0042.jpg")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import yaml
from PIL import Image

# Ultralytics base class and evaluation components
from ultralytics.engine.model import Model
from ultralytics.utils import callbacks as _ult_callbacks
from ultralytics.utils.metrics import DetMetrics, box_iou
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils import DEFAULT_CFG

# SAM 3 inference and export helpers
from .sam3_wrapper import Sam3ImageModel
from .yolo_export import YoloBox, nms_yolo_boxes, sam3_boxes_to_yolo, yolo_boxes_to_lines


# ─────────────────────────────────────────────────────────────────────────────
# Shared validator (only for match_predictions + iouv — no dataloader needed)
# ─────────────────────────────────────────────────────────────────────────────

_VALIDATOR = DetectionValidator(args=DEFAULT_CFG)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _parse_data_yaml(data: Union[str, Path, Dict]) -> Dict:
    """Parse a YOLO data.yaml, adding ``'_root'`` with the dataset root path."""
    if isinstance(data, dict):
        cfg = data.copy()
        cfg.setdefault("_root", Path("."))
        return cfg
    data_path = Path(data).resolve()
    with data_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = data_path.parent
    root = base / Path(cfg["path"]) if "path" in cfg else base
    cfg["_root"] = root.resolve()
    return cfg


def _load_yolo_label(
    label_path: Path, num_classes: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load a YOLO label file → (boxes xywhn [M,4], cls [M])."""
    boxes, classes = [], []
    if label_path.exists():
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cid = int(parts[0])
                if 0 <= cid < num_classes:
                    boxes.append(list(map(float, parts[1:5])))
                    classes.append(cid)
    if boxes:
        return (
            torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(classes, dtype=torch.long),
        )
    return torch.zeros((0, 4), dtype=torch.float32), torch.zeros(0, dtype=torch.long)


def _xywhn_to_xyxy_px(boxes: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """Normalize xywh → pixel xyxy."""
    if boxes.numel() == 0:
        return boxes
    cx, cy, bw, bh = boxes.unbind(1)
    return torch.stack(
        [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h],
        dim=1,
    )


def _yoloboxes_to_xyxy_px(boxes: List[YoloBox], w: int, h: int) -> torch.Tensor:
    """List[YoloBox] (normalized cx,cy,bw,bh) → pixel xyxy tensor."""
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    coords = [
        [(b.cx - b.w / 2) * w, (b.cy - b.h / 2) * h,
         (b.cx + b.w / 2) * w, (b.cy + b.h / 2) * h]
        for b in boxes
    ]
    return torch.tensor(coords, dtype=torch.float32)


def _sort_key(p: Path):
    try:
        return int(p.stem)
    except ValueError:
        return p.stem


# ─────────────────────────────────────────────────────────────────────────────
# Sam3Model — proper ultralytics Model subclass
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

    Examples
    --------
    >>> model = Sam3Model()
    >>> metrics = model.val(data="data/VistaSynth/data.yaml", split="val")
    >>> print(metrics.box.map50)
    >>> results = model.predict("frame_0042.jpg")
    """

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        # Bypass Model.__init__ (which expects a YOLO .pt file) and call the
        # nn.Module constructor directly, then populate all attributes that
        # the rest of the Model API expects.
        torch.nn.Module.__init__(self)

        self.callbacks = _ult_callbacks.get_default_callbacks()
        self.predictor = None
        self.model = Sam3ImageModel(checkpoint_path)   # our SAM 3 backbone
        self.trainer = None
        self.ckpt = {}
        self.cfg = None
        self.ckpt_path = str(checkpoint_path) if checkpoint_path else None
        self.overrides = {"task": "detect"}
        self.metrics = None
        self.session = None
        self.task = "detect"
        self.model_name = "sam3"

    # ── task_map required by Model._smart_load ────────────────────────────────

    @property
    def task_map(self) -> dict:
        # val() and predict() are fully overridden here, so task_map is only
        # consulted by methods we don't use (train, export, …).
        return {"detect": {"validator": DetectionValidator}}

    # ── predict ──────────────────────────────────────────────────────────────

    def predict(
        self,
        source: Union[str, Path, None] = None,
        stream: bool = False,
        predictor=None,
        **kwargs: Any,
    ) -> List[YoloBox]:
        """
        Run SAM 3 on a single image and return detected boxes.

        Parameters
        ----------
        source : str or Path
            Path to the input image (JPG / PNG).
        prompts : dict[int, str] or None
            ``{class_id: text_prompt}`` mapping.  Defaults to the ``names``
            field of the last ``val()`` call, or ``CLASS_PROMPTS`` from
            ``src.prompts``.
        conf : float
            Minimum confidence to keep a detection (default 0.0, keep all).
        iou : float
            IoU threshold for class-wise NMS (default 0.7).
        max_det : int
            Global cap on detections per image after NMS (default 300).

        Returns
        -------
        list[YoloBox]
            YOLO-format boxes after NMS (with ``.score`` and ``.features``).
        """
        if source is None:
            raise ValueError("'source' must be provided for Sam3Model.predict()")

        prompts: Optional[Dict[int, str]] = kwargs.get("prompts")
        conf: float = kwargs.get("conf", 0.0)
        iou: float = kwargs.get("iou", 0.7)
        max_det: int = kwargs.get("max_det", 300)

        if prompts is None:
            # Fall back to prompts stored from the last val() call, or defaults
            prompts = getattr(self, "_last_prompts", None)
        if prompts is None:
            from .prompts import CLASS_PROMPTS
            prompts = CLASS_PROMPTS

        img_path = Path(source)
        image = Image.open(img_path).convert("RGB")
        width, height = image.size

        all_boxes: List[YoloBox] = []
        for class_id, prompt in sorted(prompts.items()):
            pred = self.model.predict_with_text(img_path, prompt)
            all_boxes.extend(
                sam3_boxes_to_yolo(
                    prediction=pred,
                    class_id=class_id,
                    image_width=width,
                    image_height=height,
                    score_threshold=conf,
                )
            )

        return nms_yolo_boxes(all_boxes, iou_threshold=iou, max_det=max_det)

    # ── val ──────────────────────────────────────────────────────────────────

    def val(
        self,
        validator=None,
        **kwargs: Any,
    ) -> DetMetrics:
        """
        Validate SAM 3 on a full dataset split using ultralytics ``DetMetrics``.

        This method has the same signature as ``YOLO().val()`` so it drops in
        as a direct replacement.  Extra kwargs understood by Sam3Model:

        Parameters (pass as keyword arguments)
        ---------------------------------------
        data : str or Path
            YOLO-format ``data.yaml`` path.
        split : str
            ``"train"``, ``"val"``, or ``"test"`` (default ``"val"``).
        prompts : dict[int, str] or None
            ``{class_id: text_prompt}``.  Falls back to ``names`` from YAML,
            then to ``CLASS_PROMPTS``.
        conf : float
            Export confidence threshold — keep LOW (default 0.05) so that
            ``eval_conf`` can select the operating point.
        iou : float
            NMS IoU threshold (default 0.7).
        max_det : int
            Max detections per image after NMS (default 300).
        eval_conf : float
            Confidence threshold for TP/FP assignment (default 0.26).
        verbose : bool
            Print progress and final table (default True).
        save_dir : str or Path or None
            If set, prediction ``.txt`` files are written here.

        Returns
        -------
        ultralytics.utils.metrics.DetMetrics
            Same object type returned by ``YOLO().val()``.
            Access via ``metrics.box.map50``, ``metrics.box.map``,
            ``metrics.box.mp``, ``metrics.box.mr``, ``metrics.box.maps``,
            ``metrics.results_dict``.
        """
        # ── parse kwargs (mirrors ultralytics kwarg handling) ─────────────────
        data        = kwargs.get("data") or self.overrides.get("data")
        split       = kwargs.get("split", "val")
        prompts     = kwargs.get("prompts")
        conf        = kwargs.get("conf", 0.05)
        iou         = kwargs.get("iou", 0.7)
        max_det     = kwargs.get("max_det", 300)
        eval_conf   = kwargs.get("eval_conf", 0.26)
        verbose     = kwargs.get("verbose", True)
        save_dir    = kwargs.get("save_dir")

        if data is None:
            raise ValueError("Pass 'data' (path to data.yaml) to model.val().")

        # ── parse data config ─────────────────────────────────────────────────
        cfg = _parse_data_yaml(data)
        root: Path = cfg["_root"]

        if split not in cfg:
            available = [k for k in cfg if not k.startswith("_")]
            raise ValueError(
                f"Split '{split}' not found in data config. Available: {available}"
            )

        split_rel: str = cfg[split]
        images_dir = root / split_rel
        labels_dir = root / split_rel.replace("images", "labels", 1)

        if not images_dir.exists():
            raise FileNotFoundError(f"Images dir not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"Labels dir not found: {labels_dir}")

        # ── prompts & class names ─────────────────────────────────────────────
        if prompts is None:
            if "names" in cfg:
                prompts = {i: n for i, n in enumerate(cfg["names"])}
            else:
                from .prompts import CLASS_PROMPTS
                prompts = CLASS_PROMPTS
        self._last_prompts = prompts  # cache for predict()

        class_names: Dict[int, str] = dict(prompts)
        num_classes = len(prompts)

        # ── collect images ────────────────────────────────────────────────────
        image_files = sorted(
            list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")),
            key=_sort_key,
        )
        if not image_files:
            raise RuntimeError(f"No images found in {images_dir}")

        # ── optional prediction save dir ──────────────────────────────────────
        pred_dir: Optional[Path] = None
        if save_dir is not None:
            pred_dir = Path(save_dir)
            pred_dir.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"SAM 3 validation — split='{split}'")
            print(f"  Images   : {images_dir}  ({len(image_files)} files)")
            print(f"  Labels   : {labels_dir}")
            print(f"  Prompts  : {class_names}")
            print(
                f"  conf={conf}  iou={iou}  max_det={max_det}  "
                f"eval_conf={eval_conf}"
            )
            print()

        # ── ultralytics DetMetrics accumulator ────────────────────────────────
        metrics = DetMetrics(names=class_names)

        from time import time
        t0 = time()

        for idx, img_path in enumerate(image_files, start=1):
            # GT
            label_path = labels_dir / f"{img_path.stem}.txt"
            gt_boxes_xywhn, gt_cls = _load_yolo_label(label_path, num_classes)

            # SAM 3 inference
            image = Image.open(img_path).convert("RGB")
            width, height = image.size

            all_boxes: List[YoloBox] = []
            for class_id, prompt in sorted(prompts.items()):
                pred = self.model.predict_with_text(img_path, prompt)
                all_boxes.extend(
                    sam3_boxes_to_yolo(
                        prediction=pred,
                        class_id=class_id,
                        image_width=width,
                        image_height=height,
                        score_threshold=conf,
                    )
                )
            all_boxes = nms_yolo_boxes(all_boxes, iou_threshold=iou, max_det=max_det)

            # Persist predictions if requested
            if pred_dir is not None:
                lines = yolo_boxes_to_lines(all_boxes, include_score_column=True)
                with (pred_dir / f"{img_path.stem}.txt").open("w", encoding="utf-8") as f:
                    for line in lines:
                        f.write(line + "\n")

            # Apply eval_conf operating point
            kept = [b for b in all_boxes if (b.score or 0.0) >= eval_conf]

            # Tensors
            gt_xyxy = _xywhn_to_xyxy_px(gt_boxes_xywhn, width, height)
            gt_cls_np = gt_cls.cpu().numpy()

            n_pred = len(kept)
            n_gt = len(gt_xyxy)

            if n_pred > 0 and n_gt > 0:
                pred_xyxy = _yoloboxes_to_xyxy_px(kept, width, height)
                pred_conf_np = np.array([b.score or 0.0 for b in kept], dtype=np.float32)
                pred_cls_t = torch.tensor([b.class_id for b in kept], dtype=torch.long)

                # IoU [M_gt, N_pred] → match_predictions → tp [N_pred, 10]
                iou_mat = box_iou(gt_xyxy, pred_xyxy)
                tp = _VALIDATOR.match_predictions(pred_cls_t, gt_cls, iou_mat).cpu().numpy()
                pred_cls_np = pred_cls_t.cpu().numpy()

            elif n_pred > 0:
                tp = np.zeros((n_pred, _VALIDATOR.niou), dtype=bool)
                pred_conf_np = np.array([b.score or 0.0 for b in kept], dtype=np.float32)
                pred_cls_np = np.array([b.class_id for b in kept], dtype=np.float32)

            else:
                tp = np.zeros((0, _VALIDATOR.niou), dtype=bool)
                pred_conf_np = np.zeros(0, dtype=np.float32)
                pred_cls_np = np.zeros(0, dtype=np.float32)

            # Update ultralytics DetMetrics
            metrics.update_stats(
                {
                    "tp": tp,
                    "conf": pred_conf_np,
                    "pred_cls": pred_cls_np,
                    "target_cls": gt_cls_np,
                    "target_img": np.unique(gt_cls_np),
                }
            )

            if verbose:
                print(
                    f"  [{idx}/{len(image_files)}] {img_path.name}"
                    f" → {n_pred} preds, {n_gt} gt  ({time() - t0:.1f}s)"
                )

        # ── finalise ultralytics metrics ──────────────────────────────────────
        if verbose:
            print("\nFinalising metrics …")

        metrics.process()
        self.metrics = metrics  # stored for Model API compatibility

        if verbose:
            self._print_metrics(metrics, class_names)

        return metrics

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_metrics(metrics: DetMetrics, class_names: Dict[int, str]) -> None:
        """Pretty-print per-class and global metrics table."""
        header = f"\n{'Class':<20} {'GT':>6} {'AP50':>8} {'AP50-95':>10}"
        print(header)
        print("-" * len(header.lstrip()))

        nt = metrics.nt_per_class
        ap50_all = metrics.box.ap50
        ap_all = metrics.box.ap

        for i, name in sorted(class_names.items()):
            gt_cnt = int(nt[i]) if nt is not None and len(nt) > i else -1
            ap50 = float(ap50_all[i]) if ap50_all is not None and len(ap50_all) > i else 0.0
            ap = float(ap_all[i]) if ap_all is not None and len(ap_all) > i else 0.0
            print(f"  {name:<18} {gt_cnt:>6} {ap50:>8.4f} {ap:>10.4f}")

        print("-" * len(header.lstrip()))
        print(
            f"  {'all':<18} {'':>6}"
            f" {metrics.box.map50:>8.4f}"
            f" {metrics.box.map:>10.4f}"
            f"   P={metrics.box.mp:.4f}  R={metrics.box.mr:.4f}"
        )
        print()
