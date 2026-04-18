"""
VISTAValidator — shared base for all VISTA open-vocabulary model validators.

Extends ``DetectionValidator`` with:
  - Common ``__call__`` that bypasses AutoBackend (all VISTA models are HF /
    custom backends, not YOLO .pt files).
  - Structured output after each validation run:
      <save_dir>/metrics_summary.json   scalar + per-class metrics + speed block
      <save_dir>/metrics_summary.csv    same, tabular (+ speed/throughput rows)
      <save_dir>/confusion_matrix.json  matrix with class labels
      <save_dir>/pr_curves.json         precision-recall curve per class
      <save_dir>/confidence_curves.json P / R / F1 vs confidence per class
      <save_dir>/speed_summary.json     per-stage ms, FPS, totals

The saving logic lives in ``VISTAOutputMixin`` so that native-backend models
(e.g. YOLOE .pt files) can also emit the same structured outputs without
inheriting the custom ``__call__`` that bypasses AutoBackend.

Usage
-----
    # HuggingFace / custom backends:
    class MyValidator(VISTAValidator):
        def _infer(self, model, batch):
            ...  # return list[dict] in DetectionValidator format

    # Native .pt backends (YOLO family):
    class MyValidator(VISTAOutputMixin, DetectionValidator):
        pass  # structured outputs are added automatically
"""

from __future__ import annotations

import csv
import json
from abc import abstractmethod
from pathlib import Path

import numpy as np
import torch

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import TQDM, callbacks as ult_callbacks
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    from ultralytics.utils.torch_utils import unwrap_model as de_parallel


# ── structured-output mixin ───────────────────────────────────────────────────

class VISTAOutputMixin:
    """Mixin that adds structured JSON/CSV output to any DetectionValidator.

    Works with both the custom ``VISTAValidator.__call__`` flow (HF backends)
    and the standard ``DetectionValidator.__call__`` flow (native .pt models).
    All methods rely only on attributes that are present in every
    ``DetectionValidator`` subclass (``self.metrics``, ``self.speed``,
    ``self.save_dir``, ``self.names``, ``self.seen``).
    """

    # ── finalize hook ─────────────────────────────────────────────────────────

    def finalize_metrics(self) -> None:
        """Finalize metrics and save all structured outputs to ``save_dir``."""
        super().finalize_metrics()
        try:
            self._save_metrics_json()
            self._save_metrics_csv()
            self._save_confusion_matrix_json()
            self._save_pr_curves_json()
            self._save_confidence_curves_json()
            self._save_speed_json()
        except Exception as exc:  # never let serialization crash a val run
            import traceback
            from ultralytics.utils import LOGGER
            LOGGER.warning(
                f"VISTAOutputMixin: could not save structured outputs — "
                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
            )

    # ── speed helpers ─────────────────────────────────────────────────────────

    def _speed_stats(self) -> dict:
        """Return a dict with per-stage ms, derived FPS, and dataset counts."""
        sp = self.speed  # {preprocess, inference, loss, postprocess} ms/image

        preprocess_ms  = round(sp.get("preprocess",  0.0), 3)
        inference_ms   = round(sp.get("inference",   0.0), 3)
        loss_ms        = round(sp.get("loss",        0.0), 3)
        postprocess_ms = round(sp.get("postprocess", 0.0), 3)
        pipeline_ms    = round(preprocess_ms + inference_ms + postprocess_ms, 3)

        fps_inference = round(1e3 / inference_ms,  1) if inference_ms  > 0 else 0.0
        fps_pipeline  = round(1e3 / pipeline_ms,   1) if pipeline_ms   > 0 else 0.0

        total_images    = int(getattr(self, "seen", 0))
        total_instances = int(
            self.metrics.nt_per_class.sum()
            if getattr(self.metrics, "nt_per_class", None) is not None
            else 0
        )
        total_time_s = round(pipeline_ms * total_images / 1e3, 3)

        return {
            "preprocess_ms_per_image":  preprocess_ms,
            "inference_ms_per_image":   inference_ms,
            "loss_ms_per_image":        loss_ms,
            "postprocess_ms_per_image": postprocess_ms,
            "pipeline_ms_per_image":    pipeline_ms,
            "fps_inference":            fps_inference,
            "fps_pipeline":             fps_pipeline,
            "total_images":             total_images,
            "total_instances":          total_instances,
            "total_pipeline_time_s":    total_time_s,
        }

    # ── serialisation helpers ─────────────────────────────────────────────────

    def _save_metrics_json(self) -> None:
        """Save scalar summary + per-class metrics to ``metrics_summary.json``."""
        m = self.metrics

        summary_rows = m.summary() if hasattr(m, "summary") else []

        payload = {
            "summary": {
                "P":          float(np.nan_to_num(m.box.mp)),
                "R":          float(np.nan_to_num(m.box.mr)),
                "mAP50":      float(np.nan_to_num(m.box.map50)),
                "mAP50-95":   float(np.nan_to_num(m.box.map)),
                "fitness":    float(np.nan_to_num(m.fitness)),
            },
            "speed": self._speed_stats(),
            "per_class": {row["Class"]: {
                k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                for k, v in {k:v for k, v in row.items() if k != "Class"}.items()
            } for row in summary_rows},
        }

        out = self.save_dir / "metrics_summary.json"
        out.write_text(json.dumps(payload, indent=2))

    def _save_metrics_csv(self) -> None:
        """Save per-class metrics table to ``metrics_summary.csv``."""
        m = self.metrics
        box = m.box

        # Build rows: first "all", then per class
        rows = []

        # "all" row
        total_instances = int(m.nt_per_class.sum()) if m.nt_per_class is not None else 0
        rows.append({
            "class":     "all",
            "instances": total_instances,
            "P":         round(float(np.nan_to_num(box.mp)),    4),
            "R":         round(float(np.nan_to_num(box.mr)),    4),
            "F1":        round(float(np.nan_to_num(
                             np.mean(box.f1) if len(box.f1) else 0.0)), 4),
            "AP50":      round(float(np.nan_to_num(box.map50)), 4),
            "AP50-95":   round(float(np.nan_to_num(box.map)),   4),
        })

        # per-class rows
        for i, cls_idx in enumerate(box.ap_class_index):
            name = self.names.get(int(cls_idx), str(cls_idx))
            nt   = int(m.nt_per_class[cls_idx]) if m.nt_per_class is not None else 0
            rows.append({
                "class":     name,
                "instances": nt,
                "P":         round(float(np.nan_to_num(box.p[i])),   4),
                "R":         round(float(np.nan_to_num(box.r[i])),   4),
                "F1":        round(float(np.nan_to_num(box.f1[i])),  4),
                "AP50":      round(float(np.nan_to_num(box.ap50[i])),4),
                "AP50-95":   round(float(np.nan_to_num(box.ap[i])),  4),
            })

        out = self.save_dir / "metrics_summary.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        # Append speed / throughput block as a second section
        sp = self._speed_stats()
        speed_rows = [
            {"class": "# speed (ms/image)", "instances": "",   "P": "",  "R": "",  "F1": "",  "AP50": "",  "AP50-95": ""},
            {"class": "preprocess",          "instances": "",   "P": sp["preprocess_ms_per_image"],  "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "inference",           "instances": "",   "P": sp["inference_ms_per_image"],   "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "postprocess",         "instances": "",   "P": sp["postprocess_ms_per_image"], "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "pipeline",            "instances": "",   "P": sp["pipeline_ms_per_image"],    "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "# throughput (fps)",  "instances": "",   "P": "",  "R": "",  "F1": "",  "AP50": "",  "AP50-95": ""},
            {"class": "fps_inference",       "instances": "",   "P": sp["fps_inference"],  "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "fps_pipeline",        "instances": "",   "P": sp["fps_pipeline"],   "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "# dataset",           "instances": "",   "P": "",  "R": "",  "F1": "",  "AP50": "",  "AP50-95": ""},
            {"class": "total_images",        "instances": sp["total_images"],    "P": "", "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "total_instances",     "instances": sp["total_instances"], "P": "", "R": "", "F1": "", "AP50": "", "AP50-95": ""},
            {"class": "total_pipeline_time_s","instances": "",  "P": sp["total_pipeline_time_s"], "R": "", "F1": "", "AP50": "", "AP50-95": ""},
        ]
        with out.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writerows(speed_rows)

    def _save_confusion_matrix_json(self) -> None:
        """Save confusion matrix values and labels to ``confusion_matrix.json``."""
        cm = self.metrics.confusion_matrix
        if cm is None:
            return

        # Class labels — append "background" for the extra row/col in detect mode
        class_names = [self.names.get(i, str(i)) for i in range(cm.nc)]
        if cm.task == "detect":
            class_names.append("background")

        matrix = np.array(cm.matrix, dtype=float)
        payload = {
            "task":        cm.task,
            "class_names": class_names,
            "row_is":      "true_class",
            "col_is":      "predicted_class",
            "matrix":      matrix.tolist(),
            "matrix_normalized": (
                (matrix / (matrix.sum(axis=1, keepdims=True) + 1e-9))
                .round(4).tolist()
            ),
        }

        out = self.save_dir / "confusion_matrix.json"
        out.write_text(json.dumps(payload, indent=2))

    def _save_pr_curves_json(self) -> None:
        """Save per-class precision-recall curve data to ``pr_curves.json``."""
        box = self.metrics.box
        if not hasattr(box, "px") or not len(getattr(box, "prec_values", [])):
            return

        recall_axis = box.px.tolist() if hasattr(box.px, "tolist") else list(box.px)

        per_class = {}
        for i, cls_idx in enumerate(box.ap_class_index):
            if i >= len(box.prec_values):
                continue
            name = self.names.get(int(cls_idx), str(cls_idx))
            prec = box.prec_values[i]
            per_class[name] = {
                "ap50":      round(float(np.nan_to_num(box.ap50[i])), 4),
                "ap50_95":   round(float(np.nan_to_num(box.ap[i])),   4),
                "precision": [round(float(v), 4) for v in prec],
            }

        payload = {
            "x_label":   "Recall",
            "y_label":   "Precision",
            "recall":    [round(float(v), 4) for v in recall_axis],
            "per_class": per_class,
        }

        out = self.save_dir / "pr_curves.json"
        out.write_text(json.dumps(payload, indent=2))

    def _save_confidence_curves_json(self) -> None:
        """Save P / R / F1 vs confidence curves to ``confidence_curves.json``."""
        box = self.metrics.box
        if not hasattr(box, "px") or not len(getattr(box, "p_curve", [])):
            return

        conf_axis = box.px.tolist() if hasattr(box.px, "tolist") else list(box.px)

        per_class = {}
        for i, cls_idx in enumerate(box.ap_class_index):
            name = self.names.get(int(cls_idx), str(cls_idx))
            per_class[name] = {
                "precision": [round(float(v), 4) for v in box.p_curve[i]],
                "recall":    [round(float(v), 4) for v in box.r_curve[i]],
                "f1":        [round(float(v), 4) for v in box.f1_curve[i]],
            }

        payload = {
            "x_label":   "Confidence",
            "confidence": [round(float(v), 4) for v in conf_axis],
            "per_class":  per_class,
        }

        out = self.save_dir / "confidence_curves.json"
        out.write_text(json.dumps(payload, indent=2))

    def _save_speed_json(self) -> None:
        """Save throughput and timing statistics to ``speed_summary.json``."""
        out = self.save_dir / "speed_summary.json"
        out.write_text(json.dumps(self._speed_stats(), indent=2))


# ── HuggingFace / custom-backend validator ────────────────────────────────────

class VISTAValidator(VISTAOutputMixin, DetectionValidator):
    """Base validator for VISTA open-vocabulary models.

    Subclasses must implement ``_infer(model, batch)`` which runs the
    model-specific inference and returns predictions in the standard
    ``DetectionValidator`` format:
        list[dict] where each dict has keys
        ``bboxes`` (N, 4), ``conf`` (N,), ``cls`` (N,), ``extra`` (N, 0).

    Subclasses may also override ``_setup_device(model)`` when the device
    cannot be retrieved from ``model.parameters()`` directly (e.g. SAM3's
    nested architecture).
    """

    # ── device hook ──────────────────────────────────────────────────────────

    def _setup_device(self, model) -> torch.device:
        """Return the device the model runs on.

        Override this when the model is not a plain ``nn.Module``
        (e.g. when the actual weights live at ``model.model``).
        """
        return next(iter(model.parameters())).device

    # ── inference hook ───────────────────────────────────────────────────────

    @abstractmethod
    def _infer(self, model, batch) -> list[dict[str, torch.Tensor]]:
        """Run model inference on a preprocessed batch.

        Args:
            model: The backend model (HuggingFace, SAM3, …) with ``.names``
                and any other model-specific attributes attached before the
                validator is called.
            batch (dict): Standard YOLO batch produced by the dataloader.
                ``batch["img"]`` is a float32 tensor in [0, 1].

        Returns:
            list[dict]: One dict per image, each with keys
                ``bboxes`` (N, 4), ``conf`` (N,), ``cls`` (N,),
                ``extra`` (N, 0).  Coordinates must be in the letterboxed
                pixel space so they align with the GT boxes produced by
                ``DetectionValidator._prepare_batch``.
        """
        return model(batch["img"])

    # ── postprocess hook (override if NMS is not needed) ─────────────────────

    def postprocess(self, preds):
        """Return predictions unchanged (model already handles NMS)."""
        return preds

    # ── shared __call__ (bypasses AutoBackend) ────────────────────────────────

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Run validation without AutoBackend wrapping.

        Mirrors the structure of ``BaseValidator.__call__`` for the
        non-training, single-GPU path but replaces:
          - AutoBackend model loading → direct use of the provided model
          - Standard forward pass      → ``self._infer(model, batch)``
        """
        assert trainer is None, f"{type(self).__name__} does not support trainer mode."

        self.training = False
        self.device   = self._setup_device(model)
        self.args.half = False
        self.stride   = 32   # required by get_dataloader → build_yolo_dataset

        ult_callbacks.add_integration_callbacks(self)

        self.data       = check_det_dataset(self.args.data)
        self.dataloader = self.dataloader or self.get_dataloader(
            self.data.get(self.args.split), self.args.batch
        )

        self.run_callbacks("on_val_start")
        dt  = tuple(Profile(device=self.device) for _ in range(4))
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
            # dt[2] skipped — no training loss
            with dt[3]:
                preds = self.postprocess(preds)
            self.update_metrics(preds, batch)
            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        stats = self.get_stats()
        self.speed = dict(
            zip(self.speed.keys(),
                (x.t / len(self.dataloader.dataset) * 1e3 for x in dt))
        )
        self.finalize_metrics()   # calls VISTAOutputMixin.finalize_metrics
        self.print_results()
        self.run_callbacks("on_val_end")
        return stats
