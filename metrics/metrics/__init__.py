"""Convenience helpers for building a set of metrics from names/configs.

Kept here (not duplicated in framework/evaluator) since it's pure metric-
package plumbing — Phase 8's evaluator will import `build_metrics` rather
than instantiate metric classes directly.
"""
from __future__ import annotations

from framework.registry import METRIC_REGISTRY
from metrics.base import BaseMetric

from metrics import psnr_metric
from metrics import ssim_metric
from metrics import lpips_metric
from metrics import mse_metric

def build_metrics(names_or_cfgs: list) -> dict[str, BaseMetric]:
    """Build a name -> BaseMetric instance dict from a list of names or configs.

    Args:
        names_or_cfgs: Each entry is either a plain string (e.g. "psnr", using
            default params) or a dict config (e.g. {"name": "ssim", "params": {...}}).

    Returns:
        Dict keyed by metric name, in the order given.
    """
    metrics: dict[str, BaseMetric] = {}
    for entry in names_or_cfgs:
        cfg = {"name": entry} if isinstance(entry, str) else entry
        metrics[cfg["name"]] = METRIC_REGISTRY.build(cfg)
    return metrics
