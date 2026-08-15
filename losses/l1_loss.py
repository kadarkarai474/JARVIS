"""L1 (mean absolute error) loss — the standard restoration baseline.

Chosen as the default because it produces sharper results than L2/MSE
(less over-smoothing on edges) and is robust to occasional outlier pixels,
which matters here since the degradation includes noise that can produce
extreme pixel values.
"""

from __future__ import annotations

import torch

from framework.registry import LOSS_REGISTRY
from losses.base import BaseLoss


@LOSS_REGISTRY.register("l1")
class L1Loss(BaseLoss):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - target))
