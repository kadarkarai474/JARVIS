"""SSIM (Structural Similarity Index) metric.

Same core Gaussian-window formula as losses/ssim_loss.py, but this class
returns the raw similarity value directly (not `1 - SSIM`) — a metric
should be reported as-is for a benchmark table, not shaped for gradient
descent. Implemented independently in numpy/scipy (not by importing the
torch loss) so this metric has no torch dependency at all and is fully
testable now, same rationale as PSNRMetric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import convolve2d

from framework.registry import METRIC_REGISTRY
from metrics.base import BaseMetric, _to_numpy


def _gaussian_window_2d(window_size: int, sigma: float) -> np.ndarray:
    coords = np.arange(window_size, dtype=np.float64) - window_size // 2
    g = np.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return np.outer(g, g)


@METRIC_REGISTRY.register("ssim")
class SSIMMetric(BaseMetric):
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 1.0) -> None:
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self._window = _gaussian_window_2d(window_size, sigma)
        self._c1 = (0.01 * data_range) ** 2
        self._c2 = (0.03 * data_range) ** 2
        super().__init__()  # calls self.reset()

    def reset(self) -> None:
        self.values: list[float] = []

    def _ssim_single(self, pred: np.ndarray, target: np.ndarray) -> float:
        def filt(x: np.ndarray) -> np.ndarray:
            return convolve2d(x, self._window, mode="same", boundary="symm")

        mu_p = filt(pred)
        mu_t = filt(target)
        mu_p_sq, mu_t_sq, mu_pt = mu_p * mu_p, mu_t * mu_t, mu_p * mu_t

        sigma_p_sq = filt(pred * pred) - mu_p_sq
        sigma_t_sq = filt(target * target) - mu_t_sq
        sigma_pt = filt(pred * target) - mu_pt

        numerator = (2 * mu_pt + self._c1) * (2 * sigma_pt + self._c2)
        denominator = (mu_p_sq + mu_t_sq + self._c1) * (sigma_p_sq + sigma_t_sq + self._c2)
        return float((numerator / denominator).mean())

    def update(self, pred: Any, target: Any) -> None:
        p = np.squeeze(_to_numpy(pred)).astype(np.float64)
        t = np.squeeze(_to_numpy(target)).astype(np.float64)
        if p.ndim != 2 or t.ndim != 2:
            raise ValueError(
                f"SSIMMetric expects a single 2D (H, W) image per update() call "
                f"(after squeezing batch/channel dims of size 1), got shapes {p.shape}/{t.shape}. "
                "Loop over the batch and call update() once per image."
            )
        self.values.append(self._ssim_single(p, t))

    def compute(self) -> float:
        if not self.values:
            raise RuntimeError("SSIMMetric.compute() called with no samples — call update() first")
        return float(np.mean(self.values))
