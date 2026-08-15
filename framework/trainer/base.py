"""Abstract trainer interface.

This defines the shape every trainer must have; the full implementation
(optimizer stepping, AMP/BF16, gradient accumulation, EMA, DDP, checkpoint
resume) is Phase 7's job. Defining the interface now — and having Phase 7
implement against it rather than starting from scratch — is what "freeze
the architecture at Phase 4" means in practice: this contract shouldn't
need to change once real logic is filled in.

Any concrete Trainer must be constructed from a task + model + loss +
metrics + dataloaders, all obtained via the registries in
framework/registry, never imported directly — that's what keeps a new
model/task/loss swap-in from requiring changes here.
"""

from __future__ import annotations

import abc
from typing import Any

from framework.callbacks.base import CallbackList
from tasks.base import BaseTask


class BaseTrainer(abc.ABC):
    """Abstract trainer. Concrete implementation: Phase 7."""

    def __init__(
        self,
        task: BaseTask,
        model: Any,
        loss_fn: Any,
        optimizer: Any,
        train_loader: Any,
        val_loader: Any | None = None,
        scheduler: Any | None = None,
        callbacks: CallbackList | None = None,
        config: dict | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.callbacks = callbacks or CallbackList()
        self.config = config or {}

    @abc.abstractmethod
    def fit(self, num_epochs: int) -> None:
        """Run the full training loop for num_epochs. Phase 7."""
        raise NotImplementedError

    @abc.abstractmethod
    def train_one_epoch(self) -> dict[str, float]:
        """Run one training epoch, return a dict of aggregated training metrics. Phase 7."""
        raise NotImplementedError

    @abc.abstractmethod
    def validate(self) -> dict[str, float]:
        """Run one validation pass, return a dict of aggregated validation metrics. Phase 7/8."""
        raise NotImplementedError

    @abc.abstractmethod
    def save_checkpoint(self, path: str) -> None:
        """Save model/optimizer/scheduler/epoch state to `path`. Phase 7."""
        raise NotImplementedError

    @abc.abstractmethod
    def load_checkpoint(self, path: str) -> None:
        """Restore model/optimizer/scheduler/epoch state from `path`. Phase 7."""
        raise NotImplementedError
