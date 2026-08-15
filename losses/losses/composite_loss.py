"""Composite loss: weighted sum of any number of registered losses.

This is what turns "run an ablation" into a pure config change. E.g.:

    {"name": "composite", "params": {"components": [
        {"name": "l1", "weight": 1.0},
        {"name": "ssim", "weight": 0.1, "params": {"window_size": 11}},
    ]}}

Each component is resolved through LOSS_REGISTRY exactly like a standalone
loss — CompositeLoss adds no new resolution mechanism, it just calls
LOSS_REGISTRY.build() once per component. This means CompositeLoss itself
never needs to know about L1/SSIM/LPIPS/Frequency specifically, and a new
loss type automatically becomes usable inside a composite for free.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from framework.registry import LOSS_REGISTRY
from losses.base import BaseLoss


@LOSS_REGISTRY.register("composite")
class CompositeLoss(BaseLoss):
    def __init__(self, components: list[dict]) -> None:
        super().__init__()
        if not components:
            raise ValueError("CompositeLoss requires at least one component")

        self.weights: list[float] = []
        sub_losses: list[nn.Module] = []
        self.component_names: list[str] = []

        for comp_cfg in components:
            weight = float(comp_cfg.get("weight", 1.0))
            sub_loss = LOSS_REGISTRY.build(comp_cfg)
            sub_losses.append(sub_loss)
            self.weights.append(weight)
            self.component_names.append(comp_cfg["name"])

        self.sub_losses = nn.ModuleList(sub_losses)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for weight, sub_loss in zip(self.weights, self.sub_losses):
            total = total + weight * sub_loss(pred, target)
        return total

    def breakdown(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
        """Per-component (unweighted) loss values — useful for logging which
        term dominates during training, without re-running forward()."""
        return {
            name: float(sub_loss(pred, target).item())
            for name, sub_loss in zip(self.component_names, self.sub_losses)
        }
