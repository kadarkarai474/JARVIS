"""LPIPS (Learned Perceptual Image Patch Similarity) loss.

Wraps the `lpips` package. Unlike the other losses in this file, this one
needs pretrained network weights (AlexNet/VGG/SqueezeNet, depending on
`net`) which the `lpips` package downloads on first use — this requires
internet access once, in whatever environment actually trains the model
(not available in this build sandbox). We lazy-import `lpips` inside
__init__ (not at module load time) and raise a clear, actionable error if
the weights can't be fetched/found, rather than letting a cryptic
urllib/torch.hub traceback surface deep in a training loop.

LPIPS expects 3-channel RGB input normalized to [-1, 1]; our data is
grayscale [0, 1] (post `clip_unit` normalization). We replicate the single
channel to 3 channels and rescale to [-1, 1] internally so callers don't
need to think about this — the loss's public contract stays
`(pred, target) -> scalar`, same as every other registered loss.
"""

from __future__ import annotations

import torch

from framework.registry import LOSS_REGISTRY
from losses.base import BaseLoss


@LOSS_REGISTRY.register("lpips")
class LPIPSLoss(BaseLoss):
    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPSLoss requires the 'lpips' package (pip install lpips). "
                "It also needs to download pretrained weights on first use, "
                "which requires internet access."
            ) from exc

        try:
            self._lpips_net = lpips.LPIPS(net=net)
        except Exception as exc:  # noqa: BLE001 — surface as one clear, actionable message
            raise RuntimeError(
                f"Failed to initialize LPIPS with net={net!r}. This usually means the "
                "pretrained weights could not be downloaded (no internet access) or "
                "found in the local cache. Run this once with internet access to cache "
                "the weights before offline training."
            ) from exc

        for p in self._lpips_net.parameters():
            p.requires_grad_(False)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        def _to_lpips_input(x: torch.Tensor) -> torch.Tensor:
            if x.shape[1] == 1:
                x = x.repeat(1, 3, 1, 1)  # grayscale -> pseudo-RGB
            return x * 2.0 - 1.0  # [0, 1] -> [-1, 1], LPIPS's expected range

        return self._lpips_net(_to_lpips_input(pred), _to_lpips_input(target)).mean()
