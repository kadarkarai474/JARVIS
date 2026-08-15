"""LR scheduler registry. Same rationale as optim.py — schedulers need the
optimizer object as their first positional argument, so a small builder
wraps SCHEDULER_REGISTRY.get() rather than using .build() directly.
"""

from __future__ import annotations

import torch

from framework.registry import SCHEDULER_REGISTRY

SCHEDULER_REGISTRY.register("cosine")(torch.optim.lr_scheduler.CosineAnnealingLR)
SCHEDULER_REGISTRY.register("cosine_warm_restarts")(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)
SCHEDULER_REGISTRY.register("step")(torch.optim.lr_scheduler.StepLR)
SCHEDULER_REGISTRY.register("none")(torch.optim.lr_scheduler.ConstantLR)


def build_scheduler(cfg: dict | None, optimizer: torch.optim.Optimizer):
    """Build a scheduler from a registry config + optimizer.

    Args:
        cfg: {"name": "cosine", "params": {"T_max": 100, ...}}, or None/empty
            for no scheduling (returns None — caller should treat that as
            "don't call scheduler.step()").
    """
    if not cfg:
        return None
    name = cfg["name"]
    params = dict(cfg.get("params", {}) or {})
    ctor = SCHEDULER_REGISTRY.get(name)
    return ctor(optimizer, **params)
