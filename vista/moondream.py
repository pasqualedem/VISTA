from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from ultralytics.data.utils import check_det_dataset
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import TQDM, callbacks as ult_callbacks
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel


class MoonDreamValidator(DetectionValidator):
    """DetectionValidator adapted for MoonDream's HuggingFace inference API.

    Reuses all of DetectionValidator's metrics machinery (update_metrics,
    get_stats, finalize_metrics, print_results). Only the model-loading and
    inference steps are replaced so that the HuggingFace model is called via
    its encode_image / detect API instead of a standard YOLO forward pass.
    """

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Run validation, bypassing AutoBackend wrapping."""
        assert trainer is None, "MoonDreamValidator does not support trainer mode."

        # ---- Setup (mirrors BaseValidator.__call__ non-training path) ----
        self.training = False
        self.device = next(iter(model.parameters())).device
        self.args.half = False
        self.stride = 32  # required by get_dataloader → build_yolo_dataset

        ult_callbacks.add_integration_callbacks(self)

        self.data = check_det_dataset(self.args.data)
        self.dataloader = self.dataloader or self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

        # ---- Validation loop ----
        self.run_callbacks("on_val_start")
        dt = tuple(Profile(device=self.device) for _ in range(4))
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []

        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)
            with dt[1]:
                preds = self._moondream_infer(model, batch)
            # dt[2] skipped (no loss computation)
            with dt[3]:
                preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        stats = self.get_stats()
        self.speed = dict(
            zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt))
        )
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        return stats

    def _moondream_infer(self, model, batch) -> list[dict[str, torch.Tensor]]:
        """Run MoonDream on a preprocessed batch.

        Args:
            model: HuggingFace AutoModelForCausalLM with .names attribute.
            batch: Standard YOLO batch dict. batch["img"] is a float32
                tensor in [0, 1] of shape (B, 3, H, W).

        Returns:
            List of dicts – one per image – in the format expected by
            DetectionValidator.update_metrics: each dict has keys
            ``bboxes`` (N, 4), ``conf`` (N,), ``cls`` (N,), ``extra`` (N, 0),
            all as float32 tensors in the preprocessed image's pixel space.
        """
        imgs = batch["img"]  # (B, 3, H, W), float32, [0, 1]
        _, _, h, w = imgs.shape
        names: dict[int, str] = model.names

        batch_preds: list[dict[str, torch.Tensor]] = []
        for img_tensor in imgs:
            # Convert preprocessed (letterboxed) tensor back to PIL
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)

            encoded = model.encode_image(pil_img)
            bboxes, confs, clses = [], [], []
            for cls_id, cls_name in names.items():
                detections = model.detect(encoded, cls_name.strip())
                for bbox in detections.get("objects", []):
                    bboxes.append([bbox["x_min"] * w, bbox["y_min"] * h,
                                   bbox["x_max"] * w, bbox["y_max"] * h])
                    confs.append(1.0)  # MoonDream has no confidence score
                    clses.append(float(cls_id))

            n = len(bboxes)
            batch_preds.append({
                "bboxes": torch.tensor(bboxes, dtype=torch.float32, device=self.device)
                          if n else torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                "conf":   torch.tensor(confs,  dtype=torch.float32, device=self.device)
                          if n else torch.zeros(0, dtype=torch.float32, device=self.device),
                "cls":    torch.tensor(clses,  dtype=torch.float32, device=self.device)
                          if n else torch.zeros(0, dtype=torch.float32, device=self.device),
                "extra":  torch.zeros((n, 0),  dtype=torch.float32, device=self.device),
            })

        return batch_preds

    def postprocess(self, preds):
        """MoonDream produces final boxes directly – no NMS step needed."""
        return preds


class MoonDream(Model):
    """MoonDream2 open-vocabulary detection model wrapped in the Ultralytics Model interface.

    Loads a HuggingFace model and exposes the same ``predict`` / ``val`` /
    ``set_classes`` surface as YOLO / YOLOE, including standard
    ``ultralytics.engine.results.Results`` objects from predictions and full
    DetectionValidator metrics from validation.

    Examples:
        >>> md = MoonDream()
        >>> md.set_classes(["person", "ambulance"])
        >>> results = md.predict("image.jpg")
        >>> metrics = md.val(data="dataset.yaml")
        >>> print(metrics.box.map50)
    """

    def __init__(
        self,
        model_id: str = "moondream/moondream3-preview",
        revision: str = "main",
        task: str = "detect",
        verbose: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize MoonDream and load the HuggingFace model.

        Bypasses Model.__init__ (which expects a .pt / .yaml file) and
        populates the same attributes so the rest of the base-class helpers
        remain usable.
        """
        torch.nn.Module.__init__(self)

        self.callbacks = ult_callbacks.get_default_callbacks()
        self.predictor = None
        self.trainer = None
        self.ckpt = {}
        self.cfg = None
        self.ckpt_path = None
        self.overrides: dict = {"task": task, "model": model_id}
        self.metrics = None
        self.session = None
        self.task = task
        self.model_name = model_id
        self._names: dict[int, str] = {}

        if verbose:
            print(f"Loading MoonDream model '{model_id}' (revision={revision}) on {device}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=True,
            device_map={"": device},
        )

    # ------------------------------------------------------------------
    # task_map – used by Model._smart_load to pick validator / predictor
    # ------------------------------------------------------------------

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "validator": MoonDreamValidator,
            }
        }

    # ------------------------------------------------------------------
    # Class vocabulary  (mirrors YOLOWorld / YOLOE API)
    # ------------------------------------------------------------------

    @property
    def names(self) -> dict[int, str]:
        """Active detection vocabulary as ``{class_id: name}``."""
        return self._names

    @names.setter
    def names(self, value: dict[int, str]) -> None:
        self._names = value

    def set_classes(self, classes: list[str]) -> None:
        """Set the active detection vocabulary.

        Args:
            classes (list[str]): Class names to detect, e.g. ``["person", "car"]``.
        """
        self.names = {i: name for i, name in enumerate(classes)}

    # ------------------------------------------------------------------
    # Inference  →  returns standard Results objects
    # ------------------------------------------------------------------

    def predict(
        self,
        source: str | Path | Image.Image | list,
        classes: list[str] | None = None,
        **kwargs,
    ) -> list[Results]:
        """Run MoonDream detection on one or more images.

        Args:
            source: A single image path / PIL Image, or a list of those.
            classes: Class names to detect. Falls back to ``self.names``.

        Returns:
            list[Results]: Standard Ultralytics Results objects.
        """
        names = self._resolve_names(classes)
        images = self._load_images(source)

        results: list[Results] = []
        for img in images:
            orig_np = np.array(img)
            h, w = orig_np.shape[:2]
            encoded = self.model.encode_image(img)

            boxes: list[list[float]] = []
            for cls_id, cls_name in names.items():
                detections = self.model.detect(encoded, cls_name.strip())
                for bbox in detections.get("objects", []):
                    boxes.append([
                        bbox["x_min"] * w,
                        bbox["y_min"] * h,
                        bbox["x_max"] * w,
                        bbox["y_max"] * h,
                        1.0,
                        float(cls_id),
                    ])

            boxes_t = (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes
                else torch.zeros((0, 6), dtype=torch.float32)
            )
            results.append(Results(orig_np, path="", names=names, boxes=boxes_t))

        return results

    def __call__(self, source, classes: list[str] | None = None, **kwargs):
        return self.predict(source, classes, **kwargs)

    # ------------------------------------------------------------------
    # Validation  →  delegates to Model.val() via MoonDreamValidator
    # ------------------------------------------------------------------

    def val(self, **kwargs):
        """Validate using a YOLO-format dataset.

        Delegates to ``Model.val()`` which instantiates ``MoonDreamValidator``
        from the task map.  Pass ``data="path/to/dataset.yaml"`` and any
        other standard ultralytics val kwargs.

        Returns:
            DetMetrics: Standard Ultralytics detection metrics.
        """
        # MoonDreamValidator.init_metrics reads model.names; propagate here.
        self.model.names = self.names
        return super().val(**kwargs)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_names(self, classes: list[str] | None) -> dict[int, str]:
        if classes is not None:
            return {i: name for i, name in enumerate(classes)}
        if self.names:
            return self.names
        raise ValueError(
            "No class vocabulary set. Call set_classes(['name', ...]) or pass classes= to predict/val."
        )

    def _load_images(self, source) -> list[Image.Image]:
        """Normalise *source* to a list of PIL Images."""
        if isinstance(source, (str, Path)):
            return [Image.open(source).convert("RGB")]
        if isinstance(source, Image.Image):
            return [source]
        if isinstance(source, list):
            out = []
            for item in source:
                if isinstance(item, (str, Path)):
                    out.append(Image.open(item).convert("RGB"))
                elif isinstance(item, Image.Image):
                    out.append(item)
                else:
                    raise TypeError(f"Unsupported source type in list: {type(item)}")
            return out
        raise TypeError(f"Unsupported source type: {type(source)}")
