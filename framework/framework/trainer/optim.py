"""Optimizer registry.

Optimizers are just registered directly (torch.optim classes need no
wrapping) — the only reason for a separate `build_optimizer` function
instead of `OPTIMIZER_REGISTRY.build(cfg)` is that an optimizer's first
constructor argument is `model.parameters()`, not a config value, so it
can't come from `cfg["params"]` alone.
"""

from __future__ import annotations

from typing import Any, Iterable

import torch

from framework.registry import OPTIMIZER_REGISTRY

OPTIMIZER_REGISTRY.register("adamw")(torch.optim.AdamW)
OPTIMIZER_REGISTRY.register("adam")(torch.optim.Adam)
OPTIMIZER_REGISTRY.register("sgd")(torch.optim.SGD)


def build_optimizer(cfg: dict, parameters: Iterable[Any]) -> torch.optim.Optimizer:
    """Build an optimizer from a registry config + model parameters.

    Args:
        cfg: {"name": "adamw", "params": {"lr": 2e-4, "weight_decay": 0.01, ...}}
        parameters: model.parameters() (or a filtered subset thereof).
    """
    name = cfg["name"]
    params = dict(cfg.get("params", {}) or {})
    ctor = OPTIMIZER_REGISTRY.get(name)
    return ctor(parameters, **params)
