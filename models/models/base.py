"""Abstract base class for all restoration/SR models.

Every model registered under MODEL_REGISTRY (sanity CNN, NAFNet, Restormer,
SwinIR, or any future addition) must subclass this. Trainer/evaluator/
inference/benchmark code depends only on this interface (`forward`,
`scale_factor`), never on a concrete model class — that's what lets a new
model be added via "models/new_model.py + registry update + config" alone.

Requires torch (unlike tasks/base.py and utils/data_transforms.py, which
are deliberately torch-free). Execution-tested starting with the sanity
CNN in Phase 11 — this phase defines the contract, not a running model.
"""

from __future__ import annotations

import abc

import torch
import torch.nn as nn


class BaseRestorationModel(nn.Module, abc.ABC):
    """Contract: input (B, in_ch, H, W) -> output (B, out_ch, H*scale, W*scale).

    Attributes:
        scale_factor: Must match the scale_factor of whatever task this
            model is trained under (checked via
            `task.check_model_compatibility` at build time, not silently
            assumed).
    """

    def __init__(self, scale_factor: int, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the model.

        Args:
            x: (B, in_channels, H, W) float tensor.

        Returns:
            (B, out_channels, H * scale_factor, W * scale_factor) float tensor.
        """
        raise NotImplementedError

    def num_parameters(self) -> int:
        """Total trainable parameter count — used by the benchmark framework (Phase 10)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
