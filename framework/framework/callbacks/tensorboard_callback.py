"""TensorBoard logging callback — offline only, per the project's experiment
tracking spec (no external services, no internet dependency for logging).
"""

from __future__ import annotations

from typing import Any

from framework.callbacks.base import Callback
from framework.registry import CALLBACK_REGISTRY


@CALLBACK_REGISTRY.register("tensorboard")
class TensorBoardCallback(Callback):
    def __init__(self, log_dir: str) -> None:
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir)

    def on_batch_end(self, state: dict[str, Any]) -> None:
        step = state.get("global_step", 0)
        if "loss" in state:
            self.writer.add_scalar("train/loss", float(state["loss"]), step)
        if "lr" in state:
            self.writer.add_scalar("train/lr", float(state["lr"]), step)

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state.get("epoch", 0)
        for key, value in state.get("val_metrics", {}).items():
            self.writer.add_scalar(f"val/{key}", float(value), epoch)
        for key, value in state.get("train_metrics", {}).items():
            self.writer.add_scalar(f"train_epoch/{key}", float(value), epoch)

    def on_train_end(self, state: dict[str, Any]) -> None:
        self.writer.close()
