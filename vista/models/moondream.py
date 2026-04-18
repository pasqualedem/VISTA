from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.utils import callbacks as ult_callbacks

from .validator import VISTAValidator


class MoonDreamValidator(VISTAValidator):
    """VISTAValidator for MoonDream's HuggingFace inference API.

    Calls ``model.encode_image`` + ``model.detect`` on the letterboxed PIL
    image converted from the YOLO batch tensor.
    """

    def _infer(self, model, batch) -> list[dict[str, torch.Tensor]]:
        """Run MoonDream on a preprocessed batch.

        Args:
            model: HuggingFace AutoModelForCausalLM with ``.names``.
            batch: YOLO batch dict; ``batch["img"]`` is float32 in [0, 1],
                shape (B, 3, H, W).

        Returns:
            list[dict]: ``bboxes`` (N, 4), ``conf`` (N,), ``cls`` (N,),
            ``extra`` (N, 0) in letterboxed pixel space.
        """
        imgs = batch["img"]  # (B, 3, H, W), float32, [0, 1]
        _, _, h, w = imgs.shape
        names: dict[int, str] = model.names

        batch_preds: list[dict[str, torch.Tensor]] = []
        for img_tensor in imgs:
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
        names: list[str] | None = None,
    ) -> None:
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
        
        if names is not None:
            self.set_classes(names)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {"detect": {"validator": MoonDreamValidator}}

    @property
    def names(self) -> dict[int, str]:
        """Active detection vocabulary as ``{class_id: name}``."""
        return self._names

    @names.setter
    def names(self, value: dict[int, str]) -> None:
        self._names = value

    def set_classes(self, classes: list[str]) -> None:
        """Set the active detection vocabulary."""
        self.names = {i: name for i, name in enumerate(classes)}

    def predict(
        self,
        source: str | Path | Image.Image | list,
        classes: list[str] | None = None,
        **kwargs,
    ) -> list[Results]:
        """Run MoonDream detection on one or more images."""
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
                        bbox["x_min"] * w, bbox["y_min"] * h,
                        bbox["x_max"] * w, bbox["y_max"] * h,
                        1.0, float(cls_id),
                    ])

            boxes_t = (
                torch.tensor(boxes, dtype=torch.float32)
                if boxes else torch.zeros((0, 6), dtype=torch.float32)
            )
            results.append(Results(orig_np, path="", names=names, boxes=boxes_t))

        return results

    def __call__(self, source, classes: list[str] | None = None, **kwargs):
        return self.predict(source, classes, **kwargs)

    def val(self, **kwargs):
        """Validate via MoonDreamValidator. Pass ``data="dataset.yaml"``."""
        self.model.names = self.names
        return super().val(**kwargs)

    def _resolve_names(self, classes: list[str] | None) -> dict[int, str]:
        if classes is not None:
            return {i: name for i, name in enumerate(classes)}
        if self.names:
            return self.names
        raise ValueError(
            "No class vocabulary set. Call set_classes(['name', ...]) or pass classes= to predict/val."
        )

    def _load_images(self, source) -> list[Image.Image]:
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
