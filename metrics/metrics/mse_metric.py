"""Mean Squared Error metric."""

from __future__ import annotations

from typing import Any

import numpy as np

from framework.registry import METRIC_REGISTRY
from metrics.base import BaseMetric, _to_numpy


@METRIC_REGISTRY.register("mse")
class MSEMetric(BaseMetric):
    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        self.values: list[float] = []

    def update(self, pred: Any, target: Any) -> None:
        p = _to_numpy(pred).astype(np.float64)
        t = _to_numpy(target).astype(np.float64)
        self.values.append(float(np.mean((p - t) ** 2)))

    def compute(self) -> float:
        if not self.values:
            raise RuntimeError(
                "MSEMetric.compute() called with no samples — "
                "call update() first"
            )
        return float(np.mean(self.values))
