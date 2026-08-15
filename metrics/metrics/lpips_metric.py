"""LPIPS metric — evaluation-time perceptual similarity.

Same underlying `lpips` package as losses/lpips_loss.py, but used purely
for scoring (torch.no_grad(), no backward pass needed) and registered
separately under METRIC_REGISTRY since a metric and a loss have different
call contracts here (reset/update/compute vs. a single forward call) even
though they share the same pretrained network under the hood.

Same offline caveat as the LPIPS loss: needs pretrained weights downloaded
once with internet access before this can run.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from framework.registry import METRIC_REGISTRY
from metrics.base import BaseMetric


@METRIC_REGISTRY.register("lpips")
class LPIPSMetric(BaseMetric):
    def __init__(self, net: str = "alex") -> None:
        try:
            import lpips
        except ImportError as exc:
            raise ImportError(
                "LPIPSMetric requires the 'lpips' package (pip install lpips). "
                "It also needs to download pretrained weights on first use, "
                "which requires internet access."
            ) from exc

        try:
            self._lpips_net = lpips.LPIPS(net=net)
        except Exception as exc:  # noqa: BLE001 — one clear, actionable message
            raise RuntimeError(
                f"Failed to initialize LPIPS with net={net!r}. This usually means the "
                "pretrained weights could not be downloaded (no internet access) or "
                "found in the local cache. Run this once with internet access to cache "
                "the weights before offline evaluation."
            ) from exc

        for p in self._lpips_net.parameters():
            p.requires_grad_(False)
        self._lpips_net.eval()
        
        self._lpips_net = self._lpips_net.to(
            "cuda" if torch.cuda.is_available() else "cpu"
	)

        super().__init__()  # calls self.reset()

    def reset(self) -> None:
        self.values: list[float] = []

    def _to_lpips_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:  # (H, W) -> (1, 1, H, W)
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:  # (C, H, W) -> (1, C, H, W)
            x = x.unsqueeze(0)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return x * 2.0 - 1.0

    def update(self, pred: Any, target: Any) -> None:
        if not torch.is_tensor(pred):
            pred = torch.as_tensor(pred)
        if not torch.is_tensor(target):
            target = torch.as_tensor(target)

        device = pred.device
        self._lpips_net = self._lpips_net.to(device)
	
        with torch.no_grad():
            value = self._lpips_net(
                self._to_lpips_input(pred.float()), self._to_lpips_input(target.float())
            ).mean()
        self.values.append(float(value.item()))

    def compute(self) -> float:
        if not self.values:
            raise RuntimeError("LPIPSMetric.compute() called with no samples — call update() first")
        return float(np.mean(self.values))
