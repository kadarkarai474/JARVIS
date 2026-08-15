"""Global registry instances — one per pluggable component family.

Import from here, not from registry.py directly, e.g.:
    from framework.registry import MODEL_REGISTRY
"""

from framework.registry.registry import Registry, RegistryError

MODEL_REGISTRY = Registry("model")
LOSS_REGISTRY = Registry("loss")
METRIC_REGISTRY = Registry("metric")
TASK_REGISTRY = Registry("task")
OPTIMIZER_REGISTRY = Registry("optimizer")
SCHEDULER_REGISTRY = Registry("scheduler")
CALLBACK_REGISTRY = Registry("callback")

__all__ = [
    "Registry",
    "RegistryError",
    "MODEL_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "TASK_REGISTRY",
    "OPTIMIZER_REGISTRY",
    "SCHEDULER_REGISTRY",
    "CALLBACK_REGISTRY",
]
