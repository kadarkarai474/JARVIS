"""SSIM loss: 1 - Structural Similarity Index, computed with a Gaussian window.

Standard formulation (Wang et al., 2004). Implemented directly with
conv2d rather than pulling in an external SSIM package, since the whole
computation is ~30 lines and this avoids a version-compatibility surface
for something this small. The Phase 6 SSIM *metric* uses the same core
math but returns the raw similarity value (not 1 - SSIM) since a metric
should be reported as-is, not shaped for gradient descent.

Encourages structural/perceptual quality on top of L1's per-pixel
accuracy — commonly combined as `l1 + lambda * ssim` rather than used
alone, since SSIM alone tends to under-penalize small per-pixel shifts.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from framework.registry import LOSS_REGISTRY
from losses.base import BaseLoss


def _gaussian_kernel_1d(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    return g / g.sum()


def _gaussian_window(window_size: int, sigma: float, channels: int) -> torch.Tensor:
    kernel_1d = _gaussian_kernel_1d(window_size, sigma)
    kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)  # outer product -> (window, window)
    window = kernel_2d.expand(channels, 1, window_size, window_size).contiguous()
    return window


@LOSS_REGISTRY.register("ssim")
class SSIMLoss(BaseLoss):
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        channels: int = 1,
        data_range: float = 1.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.channels = channels
        self.data_range = data_range
        self.register_buffer("window", _gaussian_window(window_size, sigma, channels))
        self.c1 = (0.01 * data_range) ** 2
        self.c2 = (0.03 * data_range) ** 2

    def _ssim_map(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pad = self.window_size // 2
        window = self.window.to(dtype=pred.dtype, device=pred.device)

        mu_pred = F.conv2d(pred, window, padding=pad, groups=self.channels)
        mu_target = F.conv2d(target, window, padding=pad, groups=self.channels)

        mu_pred_sq = mu_pred * mu_pred
        mu_target_sq = mu_target * mu_target
        mu_pred_target = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred * pred, window, padding=pad, groups=self.channels) - mu_pred_sq
        sigma_target_sq = (
            F.conv2d(target * target, window, padding=pad, groups=self.channels) - mu_target_sq
        )
        sigma_pred_target = (
            F.conv2d(pred * target, window, padding=pad, groups=self.channels) - mu_pred_target
        )

        numerator = (2 * mu_pred_target + self.c1) * (2 * sigma_pred_target + self.c2)
        denominator = (mu_pred_sq + mu_target_sq + self.c1) * (sigma_pred_sq + sigma_target_sq + self.c2)
        return numerator / denominator

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ssim_map = self._ssim_map(pred, target)
        return 1.0 - ssim_map.mean()
