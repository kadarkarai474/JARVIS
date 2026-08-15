"""Abstract base class for all losses.

Concrete losses (L1, Charbonnier, SSIM, LPIPS, Frequency) are implemented
in Phase 5 and registered under LOSS_REGISTRY. This phase only defines the
shared contract so the trainer can call any registered loss identically:
`loss_fn(pred, target) -> scalar tensor`.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseLoss(nn.Module, abc.ABC):
    """Contract: (pred, target) -> scalar loss tensor, both (B, C, H, W)."""

    @abc.abstractmethod
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
