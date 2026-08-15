"""Abstract base class for all metrics.

Deliberately torch-optional: `update()` accepts either numpy arrays or
torch tensors (duck-typed via `_to_numpy`), so the accumulation logic
(reset/update/compute — needed for both per-image and dataset-average
reporting per the project spec) is unit-testable without a torch install,
same rationale as tasks/base.py and utils/data_transforms.py.

Concrete metrics (PSNR, SSIM, LPIPS) are implemented in Phase 6 and
registered under METRIC_REGISTRY. LPIPS specifically will need torch
(it's a learned metric), but PSNR/SSIM do not.
"""

from __future__ import annotations

import abc
from typing import Any


def _to_numpy(x: Any):
    """Best-effort conversion of a torch tensor or numpy array to numpy.

    Avoids importing torch at module load time — only touches tensor-like
    attributes if they're present (duck typing), so this file has zero
    hard dependency on torch being installed.
    """
    if hasattr(x, "detach"):  # torch.Tensor
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    if hasattr(x, "numpy"):
        x = x.numpy()
    return x


class BaseMetric(abc.ABC):
    """Streaming metric contract: reset() -> update(pred, target) * N -> compute().

    Supports both:
        - per-image metrics: reset(); update(pred, target); compute() per sample
        - dataset-average metrics: reset() once; update() per sample in a loop;
          compute() once at the end
    """

    def __init__(self) -> None:
        self.reset()

    @abc.abstractmethod
    def reset(self) -> None:
        """Clear any accumulated state. Called once before a fresh accumulation pass."""
        raise NotImplementedError

    @abc.abstractmethod
    def update(self, pred: Any, target: Any) -> None:
        """Accumulate one (pred, target) pair. Both (H, W) or (B, C, H, W)."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute(self) -> float:
        """Return the aggregated metric value over everything seen since reset()."""
        raise NotImplementedError
