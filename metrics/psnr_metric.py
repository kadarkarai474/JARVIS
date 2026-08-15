"""PSNR (Peak Signal-to-Noise Ratio) metric.

Torch-optional (like BaseMetric itself) since PSNR is pure arithmetic on
pixel values — no learned components — so it works identically whether
fed numpy arrays or torch tensors, and is fully unit-testable without a
torch install.

Stores every per-sample value (not just the running mean) so a later
per-image CSV report (Phase 8) can be built with zero changes to this
class — `compute()` gives the dataset average, `self.values` gives the
per-image list.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from framework.registry import METRIC_REGISTRY
from metrics.base import BaseMetric, _to_numpy


@METRIC_REGISTRY.register("psnr")
class PSNRMetric(BaseMetric):
    def __init__(self, data_range: float = 1.0) -> None:
        self.data_range = data_range
        super().__init__()  # calls self.reset()

    def reset(self) -> None:
        self.values: list[float] = []

    def update(self, pred: Any, target: Any) -> None:
        p = _to_numpy(pred).astype(np.float64)
        t = _to_numpy(target).astype(np.float64)
        mse = float(np.mean((p - t) ** 2))
        if mse == 0.0:
            value = float("inf")  # identical images -> infinite PSNR, standard convention
        else:
            value = 10.0 * math.log10((self.data_range**2) / mse)
        self.values.append(value)

    def compute(self) -> float:
        if not self.values:
            raise RuntimeError("PSNRMetric.compute() called with no samples — call update() first")
        finite = [v for v in self.values if math.isfinite(v)]
        if not finite:
            return float("inf")  # every sample was a perfect match
        return float(np.mean(finite))
