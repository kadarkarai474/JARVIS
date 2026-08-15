"""Frequency-domain loss: L1 distance between FFT magnitudes of pred and target.

Pixel-space losses (L1/Charbonnier) are dominated by low-frequency
(smooth-region) error simply because most pixels are in smooth regions.
Super-resolution specifically needs to recover high-frequency detail
(edges/texture) that a purely pixel-space loss under-weights. Comparing
2D FFT magnitude spectra directly penalizes missing high-frequency
content regardless of its (small) spatial pixel count.
"""

from __future__ import annotations

import torch

from framework.registry import LOSS_REGISTRY
from losses.base import BaseLoss


@LOSS_REGISTRY.register("frequency")
class FrequencyLoss(BaseLoss):
    def __init__(self, norm: str = "ortho") -> None:
        super().__init__()
        self.norm = norm

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_fft = torch.fft.fft2(pred, norm=self.norm)
        target_fft = torch.fft.fft2(target, norm=self.norm)
        return torch.mean(torch.abs(torch.abs(pred_fft) - torch.abs(target_fft)))
