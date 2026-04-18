"""
Sam3Trainer — fine-tunes SAM3 prompt embeddings via DetectionTrainer.

Only two small components are trainable:

* ``prompt_embeddings`` — ``nn.Embedding(num_classes, 256)``, one learnable
  vector per class.  Warm-started from the SAM3 text encoder so gradients
  refine rather than discover the signal from scratch.
* ``score_head`` — ``nn.Linear(256, 1)``, a lightweight re-ranker that
  maps prompt-conditioned ROI features to a scalar confidence score.

Everything else (SAM3 backbone, text encoder, mask decoder) stays frozen.

Training loss
-------------
For each image × class pair the trainer:
  1. Runs SAM3 with a zero prompt override (``torch.no_grad``) to get
     bbox proposals and pure ROI-pooled vision features.
  2. Adds the learnable class embedding to each ROI feature.
  3. Scores the result with ``score_head``.
  4. Assigns positive / negative labels based on IoU against GT boxes.
  5. Computes focal binary cross-entropy.

Usage
-----
    from vista.models.sam.src.sam3_trainer import Sam3Trainer

    trainer = Sam3Trainer(
        overrides={
            "data":   "data/VistaSynth/data.yaml",
            "epochs": 30,
            "batch":  4,
            "device": 0,
            "lr0":    1e-3,
            # Optional — override the default HuggingFace checkpoint:
            "sam3_checkpoint": None,
        }
    )
    trainer.train()
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.torch_utils import smart_inference_mode
from torchvision.ops import nms as torchvision_nms

from .sam3_model import Sam3Validator
from .sam3_wrapper import Sam3ImageModel


# ─────────────────────────────────────────────────────────────────────────────
# IoU helper
# ─────────────────────────────────────────────────────────────────────────────


def _box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Compute pairwise IoU between two xyxy box sets.

    Args:
        boxes1: (M, 4) pixel-space xyxy.
        boxes2: (N, 4) pixel-space xyxy.

    Returns:
        (M, N) IoU matrix.
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(0)

    ix1 = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    iy1 = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    ix2 = torch.min(boxes1[:, None, 2], boxes2[None, :, 2])
    iy2 = torch.min(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)

    union = area1[:, None] + area2[None, :] - inter
    return inter / union.clamp(min=1e-7)


# ─────────────────────────────────────────────────────────────────────────────
# Sam3PromptTuner
# ─────────────────────────────────────────────────────────────────────────────


class Sam3PromptTuner(nn.Module):
    """SAM3 backbone (frozen) + learnable per-class prompt embeddings.

    Trainable parameters
    --------------------
    prompt_embeddings : nn.Embedding(num_classes, embed_dim)
        One 256-d vector per class, warm-started from the SAM3 text encoder.
    score_head : nn.Linear(embed_dim, 1)
        Re-ranks SAM3 proposals using the prompt-conditioned ROI features.

    Forward contract
    ----------------
    * Called with a ``dict`` (training batch) → returns ``(loss, loss_items)``.
    * Called with a ``torch.Tensor`` (AMP check / eval) → returns an empty
      list so the ultralytics harness does not crash.
    """

    IOU_POS_THRESH: float = 0.50
    IOU_NEG_THRESH: float = 0.30
    FOCAL_GAMMA:    float = 2.0
    EMBED_DIM:      int   = 256

    def __init__(
        self,
        sam3: Sam3ImageModel,
        num_classes: int,
        prompts: Dict[int, str],
        embed_dim: int = EMBED_DIM,
    ) -> None:
        super().__init__()
        self.sam3        = sam3
        self.num_classes = num_classes
        self.prompts     = prompts
        self.embed_dim   = embed_dim

        # Required by DetectionTrainer.build_dataset / get_dataloader
        self.stride = torch.tensor([32.0])

        # ── freeze SAM3 ───────────────────────────────────────────────────────
        for p in self.sam3.model.parameters():
            p.requires_grad_(False)
        # SAM3's forward_grounding branches on `self.training` to compute GT
        # matching (which requires targets).  Keep it in eval mode at all times.
        self.sam3.eval()

        # ── trainable components ──────────────────────────────────────────────
        self.prompt_embeddings = nn.Embedding(num_classes, embed_dim)
        self.score_head        = nn.Linear(embed_dim, 1)
        # Satisfies ultralytics trainer's `unwrap_model(model).criterion` probe.
        self.criterion         = None

        self._init_from_text()

    # ── initialisation ────────────────────────────────────────────────────────

    def train(self, mode: bool = True) -> "Sam3PromptTuner":
        """Set training mode, but keep SAM3 locked in eval.

        The ultralytics training loop calls ``model.train()`` every epoch via
        ``_model_train()``.  Without this override that would flip SAM3's
        internal model to training mode, triggering its GT-matching code path
        (``if self.training: self._compute_matching(...)``) which crashes when
        no targets are provided.
        """
        super().train(mode)
        # Force SAM3 back to eval regardless of the requested mode.
        self.sam3.eval()
        return self

    @torch.no_grad()
    def _init_from_text(self) -> None:
        """Warm-start each prompt embedding from the SAM3 text encoder."""
        dummy = Image.new("RGB", (64, 64))
        for cls_id, text in self.prompts.items():
            try:
                state = self.sam3.processor.set_image(dummy)
                out   = self.sam3.processor.set_text_prompt(state=state, prompt=text)
                device = self.prompt_embeddings.weight.device
                dtype  = self.prompt_embeddings.weight.dtype
                vec = Sam3ImageModel._prompt_vector(
                    out.get("backbone_out", {}), device, dtype
                )
                self.prompt_embeddings.weight.data[cls_id].copy_(vec)
                LOGGER.debug(f"Sam3PromptTuner: cls {cls_id} initialised from text '{text}'.")
            except Exception as exc:
                LOGGER.warning(
                    f"Sam3PromptTuner: cls {cls_id} text-init failed "
                    f"({type(exc).__name__}: {exc}); keeping random init."
                )

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: Any) -> Any:
        """Training step when called with a batch dict; no-op otherwise."""
        if not isinstance(batch, dict):
            # AMP probe or non-training call — return a detached zero so the
            # caller does not crash on a missing return value.
            return torch.zeros(1, device=self.prompt_embeddings.weight.device)

        return self._compute_loss(batch)

    def _compute_loss(
        self, batch: Dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Focal-BCE loss over SAM3 proposals re-scored with learned embeddings.

        Args:
            batch: Standard YOLO batch dict.  Expected keys:
                ``im_file``, ``img`` (B,3,H,W), ``bboxes`` (N,4) norm-xywh
                in letterboxed space, ``cls`` (N,1), ``batch_idx`` (N,),
                ``ratio_pad`` (list of ((gain,gain),(padw,padh))).

        Returns:
            (loss, loss_items) where loss_items = [loss, n_pos, n_neg].
        """
        device    = self.prompt_embeddings.weight.device
        im_files  = batch["im_file"]
        batch_idx = batch["batch_idx"].long()

        # GT boxes: normalised xywh in letterboxed space → pixel xyxy
        lb_h = int(batch["img"].shape[2])
        lb_w = int(batch["img"].shape[3])
        scale = batch["bboxes"].new_tensor([lb_w, lb_h, lb_w, lb_h])
        gt_px_lb  = xywh2xyxy(batch["bboxes"].float()) * scale  # (N, 4) px in lb
        gt_cls    = batch["cls"].long().flatten()                 # (N,)

        # Null prompt vector: injects pure ROI features (no text conditioning)
        null_vec = torch.zeros(
            self.embed_dim,
            device=device,
            dtype=self.prompt_embeddings.weight.dtype,
        )

        losses: List[torch.Tensor] = []
        n_pos = n_neg = 0

        for si, im_file in enumerate(im_files):
            # ── reverse letterbox: lb px → original px ────────────────────────
            if "ratio_pad" in batch:
                rp    = batch["ratio_pad"][si]
                gain  = float(rp[0][0])
                padw  = float(rp[1][0])
                padh  = float(rp[1][1])
            else:
                # Fallback: approximate from image file dimensions
                pil   = Image.open(im_file)
                ow, oh = pil.size
                gain  = min(lb_w / ow, lb_h / oh)
                padw  = (lb_w - ow * gain) / 2
                padh  = (lb_h - oh * gain) / 2

            mask_gt       = batch_idx == si
            gt_px_lb_img  = gt_px_lb[mask_gt].to(device)
            gt_cls_img    = gt_cls[mask_gt].to(device)

            # lb px → original px: inv of (orig * gain + pad)
            gt_orig         = gt_px_lb_img.clone()
            gt_orig[:, [0, 2]] = (gt_px_lb_img[:, [0, 2]] - padw) / gain
            gt_orig[:, [1, 3]] = (gt_px_lb_img[:, [1, 3]] - padh) / gain

            pil_img = Image.open(im_file).convert("RGB")
            ori_w, ori_h = pil_img.size

            for cls_id, prompt in sorted(self.prompts.items()):
                # ── SAM3 inference (frozen backbone, no grad) ─────────────────
                with torch.no_grad():
                    pred = self.sam3.predict_with_prompt_override(
                        im_file, prompt, prompt_vec_override=null_vec
                    )

                if pred.boxes.shape[0] == 0:
                    continue

                # pred.features = pooled + null_vec = pure ROI features
                pooled    = pred.features.to(device)
                boxes_px  = Sam3ImageModel._ensure_boxes_pixels(
                    pred.boxes, ori_w, ori_h
                ).to(device)

                # ── apply learnable prompt embedding (grad flows here) ─────────
                p_emb    = self.prompt_embeddings(
                    torch.tensor(cls_id, device=device)
                )                                               # (256,)
                features = pooled + p_emb.unsqueeze(0)         # (P, 256)
                scores   = self.score_head(features).squeeze(-1)  # (P,)

                # ── assign positive / negative against class-specific GT ───────
                cls_mask   = gt_cls_img == cls_id
                cls_gt_px  = gt_orig[cls_mask]

                labels = self._assign_labels(boxes_px, cls_gt_px, device)
                keep   = labels >= 0
                if not keep.any():
                    continue

                loss = self._focal_bce(scores[keep], labels[keep].float())
                losses.append(loss)
                n_pos += int((labels[keep] == 1).sum())
                n_neg += int((labels[keep] == 0).sum())

        if not losses:
            zero = self.prompt_embeddings.weight.sum() * 0.0  # keeps grad graph
            return zero, torch.zeros(3, device=device)

        total_loss = torch.stack(losses).mean()
        loss_items = torch.tensor(
            [float(total_loss), float(n_pos), float(n_neg)], device=device
        )
        return total_loss, loss_items

    # ── label assignment & loss ───────────────────────────────────────────────

    @staticmethod
    def _assign_labels(
        pred_boxes: torch.Tensor,
        gt_boxes: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        """Return per-proposal labels: 1=positive, 0=negative, -1=ignore."""
        P      = pred_boxes.shape[0]
        labels = torch.full((P,), -1, dtype=torch.long, device=device)

        if gt_boxes.shape[0] == 0:
            labels[:] = 0   # no GT → all negatives
            return labels

        iou       = _box_iou_xyxy(pred_boxes, gt_boxes.to(device))
        max_iou   = iou.max(dim=1).values

        labels[max_iou >= Sam3PromptTuner.IOU_POS_THRESH] = 1
        labels[max_iou <  Sam3PromptTuner.IOU_NEG_THRESH] = 0
        return labels

    @staticmethod
    def _focal_bce(
        logits:  torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-reduced focal binary cross-entropy."""
        p   = torch.sigmoid(logits)
        ce  = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt  = torch.where(targets == 1, p, 1 - p)
        return (ce * (1 - pt) ** Sam3PromptTuner.FOCAL_GAMMA).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Sam3TunerValidator
# ─────────────────────────────────────────────────────────────────────────────


class Sam3TunerValidator(Sam3Validator):
    """Validator that scores SAM3 proposals with the learned prompt embeddings.

    The parent ``Sam3Validator._infer`` expects a ``Sam3ImageModel``.  This
    subclass overrides ``_infer`` to accept a ``Sam3PromptTuner`` and uses its
    ``prompt_embeddings`` + ``score_head`` to compute confidence scores.

    Also supports trainer-driven validation (called as ``validator(trainer)``)
    by extracting the model from the trainer before delegating to the parent
    standalone ``__call__`` path.
    """

    def __call__(self, trainer=None, model=None):
        """Support both standalone (model=...) and trainer-driven (trainer=...) calls."""
        if trainer is not None:
            model = trainer.model
        # VISTAValidator.__call__ asserts trainer is None, so always pass None.
        return super().__call__(trainer=None, model=model)

    def _setup_device(self, model: Sam3PromptTuner) -> torch.device:
        return model.prompt_embeddings.weight.device

    @smart_inference_mode()
    def _infer(
        self, model: Sam3PromptTuner, batch: Dict[str, Any]
    ) -> list[dict[str, torch.Tensor]]:
        """Run SAM3 + learned re-scoring on each image in the batch.

        Args:
            model: ``Sam3PromptTuner`` with trained ``prompt_embeddings`` and
                ``score_head``.
            batch: Standard YOLO batch with ``im_file``, ``ori_shape``,
                ``ratio_pad``.

        Returns:
            List of dicts (one per image) with keys ``bboxes`` (N,4),
            ``conf`` (N,), ``cls`` (N,), ``extra`` (N,0).  Coordinates are in
            letterboxed pixel space to align with DetectionValidator GT boxes.
        """
        device   = model.prompt_embeddings.weight.device
        conf_thr = self.args.conf
        iou_thr  = self.args.iou
        max_det  = self.args.max_det

        null_vec = torch.zeros(
            model.embed_dim,
            device=device,
            dtype=model.prompt_embeddings.weight.dtype,
        )

        batch_preds: list[dict[str, torch.Tensor]] = []

        for si, im_file in enumerate(batch["im_file"]):
            ori_h, ori_w = batch["ori_shape"][si]
            rp    = batch["ratio_pad"][si]
            gain  = float(rp[0][0])
            padw  = float(rp[1][0])
            padh  = float(rp[1][1])

            all_boxes:  List[torch.Tensor] = []
            all_scores: List[torch.Tensor] = []
            all_cls:    List[torch.Tensor] = []

            for cls_id, prompt in sorted(model.prompts.items()):
                pred = model.sam3.predict_with_prompt_override(
                    im_file, prompt, prompt_vec_override=null_vec
                )

                if pred.boxes.shape[0] == 0:
                    continue

                pooled   = pred.features.to(device)
                boxes_px = Sam3ImageModel._ensure_boxes_pixels(
                    pred.boxes, int(ori_w), int(ori_h)
                ).to(device)

                p_emb    = model.prompt_embeddings(
                    torch.tensor(cls_id, device=device)
                )
                features = pooled + p_emb.unsqueeze(0)
                scores   = torch.sigmoid(
                    model.score_head(features).squeeze(-1)
                )

                keep = scores >= conf_thr
                if not keep.any():
                    continue

                all_boxes.append(boxes_px[keep])
                all_scores.append(scores[keep])
                all_cls.append(
                    torch.full(
                        (keep.sum(),), cls_id,
                        dtype=torch.float32, device=device,
                    )
                )

            if not all_boxes:
                batch_preds.append({
                    "bboxes": torch.zeros((0, 4), device=device),
                    "conf":   torch.zeros(0,       device=device),
                    "cls":    torch.zeros(0,       device=device),
                    "extra":  torch.zeros((0, 0),  device=device),
                })
                continue

            boxes_t  = torch.cat(all_boxes,  dim=0)
            scores_t = torch.cat(all_scores, dim=0)
            cls_t    = torch.cat(all_cls,    dim=0)

            # Cross-class NMS
            keep_idx = torchvision_nms(boxes_t, scores_t, iou_thr)
            if max_det > 0:
                keep_idx = keep_idx[:max_det]

            # Project: original pixel → letterboxed pixel
            boxes_lb = boxes_t[keep_idx].clone()
            boxes_lb[:, [0, 2]] = boxes_lb[:, [0, 2]] * gain + padw
            boxes_lb[:, [1, 3]] = boxes_lb[:, [1, 3]] * gain + padh

            n = len(keep_idx)
            batch_preds.append({
                "bboxes": boxes_lb.to(self.device),
                "conf":   scores_t[keep_idx].to(self.device),
                "cls":    cls_t[keep_idx].to(self.device),
                "extra":  torch.zeros((n, 0), device=self.device),
            })

        return batch_preds


# ─────────────────────────────────────────────────────────────────────────────
# Sam3Trainer
# ─────────────────────────────────────────────────────────────────────────────


class Sam3Trainer(DetectionTrainer):
    """Trains SAM3 prompt embeddings (and a small score head) via DetectionTrainer.

    All SAM3 weights stay frozen; only ``Sam3PromptTuner.prompt_embeddings``
    and ``Sam3PromptTuner.score_head`` receive gradient updates.

    The dataloader always uses ``rect=True`` + ``augment=False`` so that
    ``ratio_pad`` is present in every training batch and GT box coordinates
    can be cleanly reversed from letterboxed to original pixel space for
    matching against SAM3's file-based inference.

    Parameters (via ``overrides`` dict)
    ------------------------------------
    data : str
        Path to the YOLO dataset YAML.
    epochs : int
        Number of training epochs (default 30).
    batch : int
        Batch size (default 4).  Keep small: SAM3 runs once per image×class.
    lr0 : float
        Initial learning rate (default 1e-3).
    device : int | str
        CUDA device index or "cpu".
    sam3_checkpoint : str | None
        Path to a local SAM3 checkpoint.  ``None`` loads from HuggingFace.
    sam3_prompts : dict[int, str] | None
        Override class prompts.  Falls back to dataset class names.

    Example
    -------
    ::

        trainer = Sam3Trainer(
            overrides={
                "data":   "data/VistaSynth/data.yaml",
                "epochs": 30,
                "batch":  4,
                "lr0":    1e-3,
            }
        )
        trainer.train()

        # Save and reload prompt embeddings
        state = torch.load(trainer.best)
        tuner.load_state_dict(state["model"])
    """

    def __init__(self, cfg=None, overrides: Optional[Dict] = None, _callbacks=None):
        from ultralytics.cfg import DEFAULT_CFG
        cfg = cfg or DEFAULT_CFG
        overrides = overrides or {}
        # Provide a dummy model name so BaseTrainer does not error on missing "model" key.
        overrides.setdefault("model", "sam3")
        # Disable mosaic / augmentation — SAM3 needs the original files.
        overrides.setdefault("mosaic", 0.0)
        overrides.setdefault("augment", False)
        super().__init__(cfg, overrides, _callbacks)
        self.loss_names = ("score",)
        # Optional: store custom prompts set before training
        self._sam3_prompts: Optional[Dict[int, str]] = overrides.get("sam3_prompts", None)

    # ── model setup ──────────────────────────────────────────────────────────

    def setup_model(self):
        """Build Sam3PromptTuner, bypassing the standard .pt-file loading."""
        if isinstance(self.model, torch.nn.Module):
            return  # already set up (e.g. resumed training)
        self.model = self.get_model(verbose=RANK in {-1, 0})

    def get_model(
        self,
        cfg=None,
        weights=None,
        verbose: bool = True,
    ) -> Sam3PromptTuner:
        """Build and return a ``Sam3PromptTuner`` for the current dataset."""
        ckpt    = getattr(self.args, "sam3_checkpoint", None)
        sam3    = Sam3ImageModel(ckpt)
        nc      = self.data["nc"]
        names   = self.data["names"]

        if self._sam3_prompts is not None:
            prompts = self._sam3_prompts
        else:
            prompts = {i: names[i] for i in range(nc)}

        tuner = Sam3PromptTuner(sam3, num_classes=nc, prompts=prompts)

        if verbose:
            n_train = sum(p.numel() for p in tuner.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in tuner.parameters())
            LOGGER.info(
                f"Sam3Trainer | trainable {n_train:,} / {n_total:,} params "
                f"({100 * n_train / max(n_total, 1):.2f} %)"
            )

        return tuner

    def set_model_attributes(self) -> None:
        """Attach dataset metadata to the tuner (nc, names, args)."""
        self.model.nc    = self.data["nc"]
        self.model.names = self.data["names"]
        self.model.args  = self.args

    # ── data ─────────────────────────────────────────────────────────────────

    def build_dataset(self, img_path: str, mode: str = "train", batch: Optional[int] = None):
        """Build a YOLO dataset with rect+no-augment so ratio_pad is available."""
        orig_aug = self.args.augment
        self.args.augment = False
        try:
            return build_yolo_dataset(
                self.args,
                img_path,
                batch,
                self.data,
                mode=mode,
                rect=True,   # letterbox only — gives ratio_pad in every item
                stride=32,
            )
        finally:
            self.args.augment = orig_aug

    def get_dataloader(
        self,
        dataset_path: str,
        batch_size: int = 16,
        rank: int = 0,
        mode: str = "train",
    ):
        """Standard YOLO dataloader (rect=True keeps ratio_pad in each batch)."""
        from ultralytics.utils.torch_utils import torch_distributed_zero_first
        assert mode in {"train", "val"}
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers,
            shuffle=shuffle,
            rank=rank,
        )

    # ── batch preprocessing ───────────────────────────────────────────────────

    def preprocess_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move label tensors to device; leave im_file untouched."""
        dev = self.device
        for key in ("bboxes", "cls", "batch_idx"):
            if key in batch and isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(dev, non_blocking=True)
        # img stays on CPU — SAM3 reads from file paths, not from the tensor
        return batch

    # ── validator ────────────────────────────────────────────────────────────

    def get_validator(self) -> Sam3TunerValidator:
        """Return a validator that uses the learned prompt embeddings."""
        self.loss_names = ("score",)
        return Sam3TunerValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=self.args,
        )

    # ── loss display ─────────────────────────────────────────────────────────

    def label_loss_items(
        self,
        loss_items: Optional[torch.Tensor] = None,
        prefix: str = "train",
    ) -> Dict[str, float]:
        keys = [f"{prefix}/{n}" for n in self.loss_names]
        if loss_items is None:
            return keys
        return {k: round(float(v), 5) for k, v in zip(keys, loss_items[:1])}

    def progress_string(self) -> str:
        return (
            "\n"
            + "%11s" * (4 + len(self.loss_names))
            % ("Epoch", "GPU_mem", *self.loss_names, "Instances", "Size")
        )

    # ── public helpers ────────────────────────────────────────────────────────

    def final_eval(self) -> None:
        """Run final validation using the in-memory Sam3PromptTuner.

        Overrides the base ``final_eval`` which passes a checkpoint *path* to
        the validator.  ``Sam3TunerValidator`` requires the live model object
        (to access ``prompt_embeddings`` and ``score_head``), so we skip the
        disk-reload step and evaluate directly.
        """
        LOGGER.info("\nRunning final validation with trained Sam3PromptTuner…")
        self.validator.args.plots = self.args.plots
        self.metrics = self.validator(model=self.model)
        self.metrics.pop("fitness", None)
        self.run_callbacks("on_fit_epoch_end")

    def set_prompts(self, prompts: Dict[int, str]) -> None:
        """Override class prompts before or during training.

        Args:
            prompts: ``{class_id: text_prompt}`` mapping.
        """
        self._sam3_prompts = prompts
        if isinstance(self.model, Sam3PromptTuner):
            self.model.prompts = prompts
