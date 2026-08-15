"""Charbonnier loss: a smooth, differentiable-everywhere approximation of L1.

sqrt((pred - target)^2 + eps^2) behaves like L2 near zero (smooth gradient,
avoids the non-differentiable kink L1 has at exactly zero error) and like
L1 for larger errors (still robust to outliers) — widely used in SR/
restoration literature (e.g. EDSR-style training) as a drop-in L1
replacement with slightly more stable optimization.
"""

from __future__ import annotations

import torch

from framework.registry import LOSS_REGISTRY
from losses.base import BaseLoss


@LOSS_REGISTRY.register("charbonnier")
class CharbonnierLoss(BaseLoss):
    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
