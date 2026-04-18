"""
Lightweight wrapper around the official SAM 3 image model.

This module centralizes all interactions with the SAM 3 API so that the rest
of the project does not depend on the low-level interface of the original
repository. This makes the code easier to maintain and test.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union, Optional

from PIL import Image
import torch
import torch.nn.functional as F  # <-- PATCH: needed for padding prompt vector

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


PathLike = Union[str, Path]


@dataclass
class Sam3Prediction:
    """Masks, bounding boxes and scores returned by SAM 3."""
    masks: Any      # usually a tensor of shape [N, 1, H, W]
    boxes: Any      # usually a tensor of shape [N, 4] in pixel coordinates
    scores: Any     # usually a tensor of shape [N]
    features: Any   # per-box semantic features (N, 256)


class Sam3ImageModel(torch.nn.Module):
    """Wrapper for single-image inference with a text prompt."""

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        """Build the SAM 3 image model and its processor."""
        # Explicitly load from HuggingFace to avoid bpe_path issues
        # NOTE: We also pass bpe_path explicitly to be robust in Colab/runtime environments.
        import sam3  # local import to avoid issues before dependencies are ready
        
        super().__init__()

        bpe_path = str(Path("assets") / "bpe_simple_vocab_16e6.txt.gz")
        if not os.path.exists(bpe_path):
            url = "https://github.com/openai/CLIP/raw/refs/heads/main/clip/bpe_simple_vocab_16e6.txt.gz"
            os.makedirs(os.path.dirname(bpe_path), exist_ok=True)
            import wget
            print(f"Downloading BPE vocab from {url} to {bpe_path}...")
            wget.download(url, bpe_path)

        if checkpoint_path is not None:
            # Load from explicit checkpoint path
            self.model = build_sam3_image_model(checkpoint_path=checkpoint_path, bpe_path=bpe_path)
        else:
            # Load from HuggingFace
            self.model = build_sam3_image_model(load_from_HF=True, bpe_path=bpe_path)
        self.processor = Sam3Processor(self.model)

    @staticmethod
    def _pick_feature_map(backbone_out: Dict[str, Any]) -> torch.Tensor:
        """
        Pick a reasonable vision feature map for ROI pooling.
        Priority:
          1) backbone_fpn[0] (highest resolution pyramid feature)
          2) vision_features
          3) sam2_backbone_out (if tensor-like)
        Returns a tensor shaped [1, C, H, W].
        """
        cand = None

        if isinstance(backbone_out.get("backbone_fpn", None), (list, tuple)) and backbone_out["backbone_fpn"]:
            cand = backbone_out["backbone_fpn"][0]
        elif isinstance(backbone_out.get("vision_features", None), (list, tuple)) and backbone_out["vision_features"]:
            cand = backbone_out["vision_features"][0]
        elif isinstance(backbone_out.get("vision_features", None), torch.Tensor):
            cand = backbone_out["vision_features"]
        elif isinstance(backbone_out.get("sam2_backbone_out", None), torch.Tensor):
            cand = backbone_out["sam2_backbone_out"]

        if cand is None or not isinstance(cand, torch.Tensor):
            raise RuntimeError(
                "Could not find a usable vision feature map inside output['backbone_out']. "
                f"Available keys: {list(backbone_out.keys())}"
            )

        # Normalize shape to [1, C, H, W]
        if cand.ndim == 3:
            cand = cand.unsqueeze(0)
        elif cand.ndim != 4:
            raise RuntimeError(f"Unexpected feature map shape {tuple(cand.shape)}; expected 3D or 4D tensor.")

        return cand

    @staticmethod
    def _prompt_vector(backbone_out: Dict[str, Any], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Build a 256-d prompt-conditioned vector from language embeddings.
        We use mean pooling across tokens and broadcast it to boxes.
        """
        lang = backbone_out.get("language_embeds", None)
        if not isinstance(lang, torch.Tensor):
            # Fallback: zero vector (still works, just less prompt-conditioned)
            return torch.zeros((256,), device=device, dtype=dtype)

        # Handle common shapes:
        #   (1, T, D) -> (T, D)
        if lang.ndim == 3 and lang.shape[0] == 1:
            lang = lang[0]

        #   (T, 1, D) -> (T, D)
        if lang.ndim == 3 and lang.shape[1] == 1:
            lang = lang[:, 0, :]

        # Pool tokens -> vector
        #   (T, D) -> (D,)
        #   (D,)   -> (D,)
        #   other  -> flatten token dims then pool
        if lang.ndim == 2:
            vec = lang.mean(dim=0)
        elif lang.ndim == 1:
            vec = lang
        else:
            vec = lang.reshape(-1, lang.shape[-1]).mean(dim=0)

        # Force to 256-d (truncate / pad)
        d = int(vec.numel())
        if d > 256:
            vec = vec[:256]
        elif d < 256:
            vec = F.pad(vec, (0, 256 - d))

        return vec.to(device=device, dtype=dtype)

    @staticmethod
    def _ensure_boxes_pixels(boxes: torch.Tensor, orig_w: int, orig_h: int) -> torch.Tensor:
        """
        Ensure boxes are in pixel XYXY.
        If boxes look normalized (max <= ~1.5), scale to pixels.
        """
        if boxes.numel() == 0:
            return boxes

        mx = float(boxes.max().detach().cpu().item())
        if mx <= 1.5:
            scale = boxes.new_tensor([orig_w, orig_h, orig_w, orig_h])
            return boxes * scale
        return boxes

    def fuse(self, prompt_tuner: "Sam3PromptTuner") -> "Sam3ImageModel":
        """Absorb trained prompt embeddings and score head into this model.

        After fusing, ``predict_with_text`` uses the learned per-class
        embedding (looked up by prompt string) instead of re-encoding text
        through the SAM3 language encoder.  The text is still forwarded to
        SAM3's grounding decoder for box/mask generation; only the
        ``features = pooled + prompt_vec`` conditioning and the confidence
        re-scoring use the baked-in weights.

        This mirrors the YOLOE pattern: learnable components are collapsed
        into static buffers so that inference needs no external embedding
        state and no active gradient context.

        Args:
            prompt_tuner: Trained ``Sam3PromptTuner``.  Its
                ``prompt_embeddings.weight`` and ``score_head`` are deep-copied
                and stored as frozen attributes.

        Returns:
            ``self`` for chaining.
        """
        import copy as _copy

        # ── bake embeddings as a frozen buffer ────────────────────────────────
        weight = prompt_tuner.prompt_embeddings.weight.detach().clone()
        # Register as a proper nn.Module buffer so it moves with .to(device)
        # and is included in state_dict().
        self.register_buffer("_fused_embeddings", weight)  # (nc, embed_dim)

        # ── bake score head ───────────────────────────────────────────────────
        self._fused_score_head = _copy.deepcopy(prompt_tuner.score_head)
        for p in self._fused_score_head.parameters():
            p.requires_grad_(False)

        # ── reverse lookup: text prompt → class id ────────────────────────────
        self._fused_prompt_to_cls: dict[str, int] = {
            text: cls_id for cls_id, text in prompt_tuner.prompts.items()
        }

        self._is_fused: bool = True

        from ultralytics.utils import LOGGER as _LOGGER
        _LOGGER.info(
            f"Sam3ImageModel fused: {len(self._fused_prompt_to_cls)} class "
            "embeddings absorbed — text encoder bypassed for feature conditioning."
        )
        return self

    def predict_with_text(self, image_path: PathLike, prompt: str) -> Sam3Prediction:
        """Run SAM 3 on a single image using a text prompt.

        When the model has been fused via ``fuse()``, the learned per-class
        embedding is used for feature conditioning and confidence re-scoring
        instead of the text encoder output.  Box/mask generation always uses
        the text grounding path.
        """
        if getattr(self, "_is_fused", False) and prompt in self._fused_prompt_to_cls:
            cls_id  = self._fused_prompt_to_cls[prompt]
            emb_vec = self._fused_embeddings[cls_id]  # (embed_dim,)
            pred    = self.predict_with_prompt_override(
                image_path, prompt, prompt_vec_override=emb_vec
            )
            # Re-score proposals with the baked-in score head
            if pred.features.shape[0] > 0:
                dev    = pred.features.device
                scores = torch.sigmoid(
                    self._fused_score_head.to(dev)(pred.features)
                ).squeeze(-1)
                pred = Sam3Prediction(
                    masks=pred.masks,
                    boxes=pred.boxes,
                    scores=scores,
                    features=pred.features,
                )
            return pred

        return self.predict_with_prompt_override(image_path, prompt, prompt_vec_override=None)

    def predict_with_prompt_override(
        self,
        image_path: PathLike,
        prompt: str,
        prompt_vec_override: Optional[torch.Tensor] = None,
    ) -> Sam3Prediction:
        """Run SAM 3 on a single image, optionally overriding the prompt vector.

        Identical to ``predict_with_text`` except that when *prompt_vec_override*
        is provided it replaces the text-derived ``_prompt_vector`` in the
        ``features = pooled + prompt_vec`` step.  This allows callers to inject a
        learnable embedding (with an active gradient) while keeping the SAM3
        backbone frozen under ``torch.no_grad()``.

        If *prompt_vec_override* is ``None`` the behaviour is identical to
        ``predict_with_text``.  Pass ``torch.zeros(embed_dim)`` to obtain the raw
        ROI-pooled features without any prompt conditioning.

        Args:
            image_path: Path to the input image.
            prompt: Text prompt forwarded to the SAM3 text encoder (used for
                box/mask generation regardless of *prompt_vec_override*).
            prompt_vec_override: Optional 1-D tensor of shape ``(embed_dim,)``
                substituted for the text-derived prompt vector.  When provided,
                gradients flow through this tensor.

        Returns:
            ``Sam3Prediction`` with ``features`` computed as
            ``pooled + prompt_vec_override`` (or ``pooled + text_vec`` when
            *prompt_vec_override* is ``None``).
        """
        image_path = Path(image_path)
        image = Image.open(image_path).convert("RGB")

        debug = os.environ.get("SAM3_DEBUG", "0") == "1"

        state = self.processor.set_image(image)
        output: Dict[str, Any] = self.processor.set_text_prompt(
            state=state,
            prompt=prompt,
        )

        if debug:
            print("TOP KEYS:", list(output.keys()))
            bo = output.get("backbone_out", None)
            print("backbone_out type:", type(bo))
            if isinstance(bo, dict):
                print("BACKBONE_OUT KEYS:", list(bo.keys()))
            if "boxes" in output and hasattr(output["boxes"], "shape"):
                print("boxes shape:", output["boxes"].shape)

        # Get final predictions
        final_masks = output["masks"]
        final_boxes = output["boxes"]
        final_scores = output["scores"]

        # If no detections, return empty features.
        if hasattr(final_boxes, "shape") and final_boxes.shape[0] == 0:
            empty_feats = torch.empty((0, 256), device=final_scores.device, dtype=final_scores.dtype)
            return Sam3Prediction(
                masks=final_masks,
                boxes=final_boxes,
                scores=final_scores,
                features=empty_feats,
            )

        # Build per-box semantic features via ROI pooling on a backbone feature map,
        # then condition them on the prompt using language_embeds.
        backbone_out = output.get("backbone_out", None)
        if not isinstance(backbone_out, dict):
            raise RuntimeError("SAM3 output does not contain 'backbone_out' dict; cannot extract features.")

        feat_map = self._pick_feature_map(backbone_out)  # [1, C, Hf, Wf]
        feat_map = feat_map.to(device=final_boxes.device)

        orig_h = int(output["original_height"])
        orig_w = int(output["original_width"])

        boxes_px = self._ensure_boxes_pixels(final_boxes, orig_w, orig_h)

        # ROIAlign expects boxes with batch indices: (N, 5) -> [batch_idx, x1, y1, x2, y2]
        n = boxes_px.shape[0]
        batch_idx = torch.zeros((n, 1), device=boxes_px.device, dtype=boxes_px.dtype)
        rois = torch.cat([batch_idx, boxes_px], dim=1)

        # Compute spatial_scale: feature_map pixels per input pixel
        # Usually Hf/H == Wf/W; use width-based scale.
        spatial_scale = feat_map.shape[-1] / float(orig_w)

        try:
            from torchvision.ops import roi_align
        except Exception as e:
            raise RuntimeError(
                "torchvision.ops.roi_align is required to extract per-box features. "
                "Ensure torchvision is installed and compatible with your torch build."
            ) from e

        # Pool to 1x1 per ROI -> [N, C, 1, 1] -> [N, C]
        pooled = roi_align(
            input=feat_map,
            boxes=rois,
            output_size=(1, 1),
            spatial_scale=spatial_scale,
            sampling_ratio=-1,
            aligned=True,
        ).flatten(1)

        # Prompt conditioning: use override if provided, else derive from language_embeds.
        if prompt_vec_override is not None:
            prompt_vec = prompt_vec_override.to(device=pooled.device, dtype=pooled.dtype)
        else:
            prompt_vec = self._prompt_vector(backbone_out, device=pooled.device, dtype=pooled.dtype)

        if pooled.shape[1] != prompt_vec.shape[0]:
            # If C != 256, we can't keep 256-d features. Fail loudly (avoids silent bugs).
            raise RuntimeError(
                f"Feature map channel dim C={pooled.shape[1]} does not match prompt dim {prompt_vec.shape[0]}. "
                "Pick a different feature map (expected C=256) or change downstream feature_dim."
            )

        features = pooled + prompt_vec.unsqueeze(0)  # [N, 256]

        return Sam3Prediction(
            masks=final_masks,
            boxes=final_boxes,
            scores=final_scores,
            features=features,
        )
