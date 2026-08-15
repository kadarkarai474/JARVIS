"""Memory monitoring callback.

Logs peak/current GPU memory usage into `state["memory_log"]` each epoch —
required by the project's training-features spec, and directly useful on
a 4GB RTX3050 dev card where hitting the memory ceiling is the most likely
failure mode. No-ops cleanly (doesn't crash) when CUDA isn't available,
e.g. CPU-only debugging.
"""

from __future__ import annotations

from typing import Any

from framework.callbacks.base import Callback
from framework.registry import CALLBACK_REGISTRY


@CALLBACK_REGISTRY.register("memory_monitor")
class MemoryMonitorCallback(Callback):
    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.history: list[dict[str, Any]] = []

    def _snapshot(self) -> dict[str, float] | None:
        try:
            import torch
        except ImportError:
            return None
        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated(self.device_index) / (1024**2),
            "reserved_mb": torch.cuda.memory_reserved(self.device_index) / (1024**2),
            "max_allocated_mb": torch.cuda.max_memory_allocated(self.device_index) / (1024**2),
        }

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        snap = self._snapshot()
        if snap is None:
            return
        snap["epoch"] = state.get("epoch", -1)
        self.history.append(snap)
        state.setdefault("memory_log", []).append(snap)
