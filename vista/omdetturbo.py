from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, OmDetTurboForObjectDetection

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


class OmDetTurboValidator(DetectionValidator):
    """DetectionValidator adapted for OmDetTurbo's HuggingFace inference API.

    Reuses all of DetectionValidator's metrics machinery. Only the
    model-loading and inference steps are replaced so that the HuggingFace
    model is called via its processor + forward pass instead of a standard
    YOLO forward pass.
    """

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Run validation, bypassing AutoBackend wrapping."""
        assert trainer is None, "OmDetTurboValidator does not support trainer mode."

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
                preds = self._infer(model, batch)
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

    def _infer(self, model, batch) -> list[dict[str, torch.Tensor]]:
        """Run OmDetTurbo on a preprocessed batch.

        Args:
            model: HuggingFace OmDetTurboForObjectDetection with ``.names``,
                ``._processor``, and ``._task_prompt`` attributes attached by
                ``OmDetTurbo.val()``.
            batch: Standard YOLO batch dict. ``batch["img"]`` is a float32
                tensor in [0, 1] of shape (B, 3, H, W).

        Returns:
            List of dicts – one per image – in the format expected by
            ``DetectionValidator.update_metrics``: each dict has keys
            ``bboxes`` (N, 4), ``conf`` (N,), ``cls`` (N,), ``extra`` (N, 0),
            all as float32 tensors.  Coordinates are in the preprocessed
            (letterboxed) image pixel space so they align with the GT boxes
            that ``_prepare_batch`` computes.
        """
        imgs = batch["img"]  # (B, 3, H, W), float32, [0, 1]
        _, _, h, w = imgs.shape
        names: dict[int, str] = model.names
        class_list = list(names.values())
        name_to_id = {v: k for k, v in names.items()}
        task_prompt = getattr(model, "_task_prompt", "")

        pil_images = [
            Image.fromarray((t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
            for t in imgs
        ]

        processor = model._processor
        inputs = processor(
            images=pil_images,
            text=[class_list] * len(pil_images),
            task=[task_prompt] * len(pil_images),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = model(**inputs)

        batch_results = processor.post_process_grounded_object_detection(
            outputs,
            text_labels=[class_list] * len(pil_images),
            target_sizes=[(h, w)] * len(pil_images),
            threshold=self.args.conf,
            nms_threshold=self.args.iou,
        )

        preds: list[dict[str, torch.Tensor]] = []
        for res in batch_results:
            boxes = res.get("boxes")    # (N, 4) xyxy in (h, w) pixel space
            scores = res.get("scores")  # (N,)
            labels = res.get("labels", [])
            n = len(labels)

            if n == 0:
                preds.append({
                    "bboxes": torch.zeros((0, 4), dtype=torch.float32, device=self.device),
                    "conf":   torch.zeros(0,       dtype=torch.float32, device=self.device),
                    "cls":    torch.zeros(0,       dtype=torch.float32, device=self.device),
                    "extra":  torch.zeros((0, 0),  dtype=torch.float32, device=self.device),
                })
            else:
                cls_ids = torch.tensor(
                    [float(name_to_id.get(lbl, 0)) for lbl in labels],
                    dtype=torch.float32, device=self.device,
                )
                preds.append({
                    "bboxes": boxes.to(dtype=torch.float32, device=self.device),
                    "conf":   scores.to(dtype=torch.float32, device=self.device),
                    "cls":    cls_ids,
                    "extra":  torch.zeros((n, 0), dtype=torch.float32, device=self.device),
                })

        return preds

    def postprocess(self, preds):
        """OmDetTurbo already applies NMS in post_process; return as-is."""
        return preds


class OmDetTurbo(Model):
    """OmDetTurbo open-vocabulary detection model wrapped in the Ultralytics Model interface.

    Loads ``omlab/omdet-turbo-swin-tiny-hf`` (or any compatible checkpoint)
    via HuggingFace and exposes the same ``predict`` / ``val`` / ``set_classes``
    surface as YOLO / YOLOE, including standard
    ``ultralytics.engine.results.Results`` objects from predictions and full
    ``DetectionValidator`` metrics from validation.

    Attributes:
        names (dict[int, str]): Active class vocabulary set via ``set_classes``.
        _task_prompt (str): Optional free-text task description passed to the
            model to steer grounded detection.

    Examples:
        >>> od = OmDetTurbo()
        >>> od.set_classes(["person", "ambulance"])
        >>> results = od.predict("image.jpg")
        >>> metrics = od.val(data="dataset.yaml")
        >>> print(metrics.box.map50)
    """

    def __init__(
        self,
        model_id: str = "omlab/omdet-turbo-swin-tiny-hf",
        task: str = "detect",
        verbose: bool = False,
        device: str = "cuda",
    ) -> None:
        """Initialize OmDetTurbo and load the HuggingFace model + processor.

        Bypasses ``Model.__init__`` (which expects a .pt / .yaml file) and
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
        self._task_prompt: str = ""

        if verbose:
            print(f"Loading OmDetTurbo model '{model_id}' on {device}")

        self._processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        hf_model = OmDetTurboForObjectDetection.from_pretrained(model_id, low_cpu_mem_usage=False)
        # Some buffers may still be meta tensors after from_pretrained; replace
        # them with zero-filled CPU tensors so that .to(device) works cleanly.
        for module in hf_model.modules():
            for buf_name, buf in list(module.named_buffers(recurse=False)):
                if buf is not None and buf.is_meta:
                    module.register_buffer(
                        buf_name, torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
                    )
        self.model = hf_model.to(device)

    # ------------------------------------------------------------------
    # task_map – used by Model._smart_load to pick validator
    # ------------------------------------------------------------------

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        return {
            "detect": {
                "validator": OmDetTurboValidator,
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

    def set_classes(self, classes: list[str], task_prompt: str = "") -> None:
        """Set the active detection vocabulary and optional task description.

        Args:
            classes (list[str]): Class names to detect, e.g. ``["person", "car"]``.
            task_prompt (str): Free-text task description to steer the model,
                e.g. ``"Detect vehicles on the road."``.
        """
        self.names = {i: name for i, name in enumerate(classes)}
        self._task_prompt = task_prompt

    # ------------------------------------------------------------------
    # Inference  →  returns standard Results objects
    # ------------------------------------------------------------------

    def predict(
        self,
        source: str | Path | Image.Image | list,
        classes: list[str] | None = None,
        task_prompt: str | None = None,
        conf: float = 0.10,
        iou: float = 0.3,
        **kwargs,
    ) -> list[Results]:
        """Run OmDetTurbo detection on one or more images.

        Args:
            source: A single image path / PIL Image, or a list of those.
            classes: Class names to detect. Falls back to ``self.names``.
            task_prompt: Task description. Falls back to ``self._task_prompt``.
            conf: Confidence threshold for post-processing.
            iou: NMS IoU threshold for post-processing.

        Returns:
            list[Results]: Standard Ultralytics Results objects.
        """
        names = self._resolve_names(classes)
        class_list = list(names.values())
        name_to_id = {v: k for k, v in names.items()}
        task = task_prompt if task_prompt is not None else self._task_prompt
        images = self._load_images(source)

        device = next(iter(self.model.parameters())).device
        inputs = self._processor(
            images=images,
            text=[class_list] * len(images),
            task=[task] * len(images),
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        batch_results = self._processor.post_process_grounded_object_detection(
            outputs,
            text_labels=[class_list] * len(images),
            target_sizes=[img.size[::-1] for img in images],  # (H, W) per image
            threshold=conf,
            nms_threshold=iou,
        )

        results: list[Results] = []
        for img, res in zip(images, batch_results):
            orig_np = np.array(img)
            boxes = res.get("boxes")    # (N, 4) xyxy in original pixel space
            scores = res.get("scores")  # (N,)
            labels = res.get("labels", [])
            n = len(labels)

            if n > 0:
                cls_ids = torch.tensor(
                    [float(name_to_id.get(lbl, 0)) for lbl in labels],
                    dtype=torch.float32,
                )
                boxes_t = torch.cat(
                    [boxes.cpu(), scores.cpu().unsqueeze(1), cls_ids.unsqueeze(1)], dim=1
                )
            else:
                boxes_t = torch.zeros((0, 6), dtype=torch.float32)

            results.append(Results(orig_np, path="", names=names, boxes=boxes_t))

        return results

    def __call__(self, source, classes: list[str] | None = None, **kwargs):
        return self.predict(source, classes, **kwargs)

    # ------------------------------------------------------------------
    # Validation  →  delegates to Model.val() via OmDetTurboValidator
    # ------------------------------------------------------------------

    def val(self, **kwargs):
        """Validate using a YOLO-format dataset.

        Delegates to ``Model.val()`` which instantiates ``OmDetTurboValidator``
        from the task map.  Pass ``data="path/to/dataset.yaml"`` and any
        other standard ultralytics val kwargs.

        Returns:
            DetMetrics: Standard Ultralytics detection metrics.
        """
        # OmDetTurboValidator._infer reads these attributes from the model.
        self.model.names = self.names
        self.model._processor = self._processor
        self.model._task_prompt = self._task_prompt
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
