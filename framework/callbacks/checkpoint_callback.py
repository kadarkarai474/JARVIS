"""Checkpoint callback: saves `last.pt` periodically and `best.pt` whenever
the monitored metric improves. Never overwrites past runs — this callback
operates within a single already-unique `experiments/<model>/run_NNN/`
directory (that uniqueness is the launcher script's job, per Phase 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framework.callbacks.base import Callback
from framework.registry import CALLBACK_REGISTRY
from framework.trainer.checkpoint import save_checkpoint


@CALLBACK_REGISTRY.register("checkpoint")
class CheckpointCallback(Callback):
    def __init__(
        self,
        save_dir: str,
        monitor: str = "val_psnr",
        mode: str = "max",
        save_every_n_epochs: int = 1,
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.save_dir = Path(save_dir)
        self.monitor = monitor
        self.mode = mode
        self.save_every_n_epochs = save_every_n_epochs
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def _is_improvement(self, value: float) -> bool:
        return value < self.best_value if self.mode == "min" else value > self.best_value

    def on_epoch_end(self, state: dict[str, Any]) -> None:
        epoch = state["epoch"]

        common_kwargs = dict(
            model=state["model"],
            optimizer=state.get("optimizer"),
            scheduler=state.get("scheduler"),
            scaler=state.get("precision_manager"),  # PrecisionManager duck-types .state_dict()
            ema=state.get("ema"),
            epoch=epoch,
            global_step=state.get("global_step", 0),
            config=state.get("config"),
            seed=state.get("seed"),
        )

        if (epoch + 1) % self.save_every_n_epochs == 0:
            save_checkpoint(
                self.save_dir / "last.pt", best_metric=self.best_value, **common_kwargs
            )

        val_metrics = state.get("val_metrics", {})
        if self.monitor in val_metrics:
            value = val_metrics[self.monitor]
            if self._is_improvement(value):
                self.best_value = value
                save_checkpoint(self.save_dir / "best.pt", best_metric=value, **common_kwargs)
