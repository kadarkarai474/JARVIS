"""Callback system for the training engine (Phase 7 wires these into the loop).

Pure Python, no torch dependency — fully testable now. Concrete callbacks
(checkpointing, EMA, memory monitoring, TensorBoard logging) are implemented
alongside the training engine in Phase 7 and registered under
CALLBACK_REGISTRY; this phase only defines the lifecycle contract and the
container that dispatches to a list of callbacks.
"""

from __future__ import annotations

import abc
from typing import Any


class Callback(abc.ABC):
    """Lifecycle hooks a callback may override. All are no-ops by default.

    `state` passed to each hook is a plain dict the trainer maintains
    (epoch, step, loss, metrics, model, optimizer, etc.) — kept as an
    untyped dict rather than a fixed dataclass so future trainer additions
    never require changing this interface.
    """

    def on_train_start(self, state: dict[str, Any]) -> None: ...

    def on_epoch_start(self, state: dict[str, Any]) -> None: ...

    def on_batch_end(self, state: dict[str, Any]) -> None: ...

    def on_epoch_end(self, state: dict[str, Any]) -> None: ...

    def on_train_end(self, state: dict[str, Any]) -> None: ...


class CallbackList:
    """Dispatches each lifecycle hook to every callback in registration order."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        self.callbacks: list[Callback] = list(callbacks) if callbacks else []

    def append(self, callback: Callback) -> None:
        self.callbacks.append(callback)

    def on_train_start(self, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_start(state)

    def on_epoch_start(self, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_start(state)

    def on_batch_end(self, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_batch_end(state)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_epoch_end(state)

    def on_train_end(self, state: dict[str, Any]) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def __len__(self) -> int:
        return len(self.callbacks)
