"""Concrete Trainer: the full implementation of BaseTrainer (Phase 4).

RTX3050 dev strategy (per project hardware plan): precision="fp32",
grad_accum_steps=1, use_ema=False, use_ddp=False are all safe defaults —
enable AMP/EMA/grad-accum only after this configuration is proven correct
end-to-end (that proof is Phase 11's job, per the Phase Priority Rule).

DDP note: this class does NOT set up torch.distributed itself (no
process-group init, no rank/world_size handling) — that's a launcher-script
concern (`torchrun ...`). Trainer just accepts an already-DDP-wrapped
model; if `use_ddp=True`, callback/checkpoint logic is guarded to only
act on rank 0 (via `is_main_process`) so multi-process runs don't
corrupt shared files by writing from every rank at once.
"""

from __future__ import annotations

from typing import Any

from framework.callbacks.base import CallbackList
from framework.trainer.base import BaseTrainer
from framework.trainer.checkpoint import load_checkpoint, save_checkpoint
from framework.trainer.ema import EMA
from framework.trainer.precision import PrecisionManager
from framework.trainer.step import compute_step
from metrics import build_metrics
from tasks.base import BaseTask


def _move_batch_to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


class Trainer(BaseTrainer):
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
        device: str = "cuda",
        precision: str = "fp32",
        grad_accum_steps: int = 1,
        grad_clip_norm: float | None = None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        metric_names: list | None = None,
        seed: int = 0,
        is_main_process: bool = True,
    ) -> None:
        super().__init__(task, model, loss_fn, optimizer, train_loader, val_loader, scheduler, callbacks, config)

        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

        # Fail fast on a mismatched task/model pairing rather than silently
        # producing wrong-shaped output deep in the first forward pass.
        task.check_model_compatibility(model.scale_factor)

        self.device = device
        self.precision_manager = PrecisionManager(
            precision, device_type="cuda" if "cuda" in device else "cpu"
        )
        self.grad_accum_steps = grad_accum_steps
        self.grad_clip_norm = grad_clip_norm
        self.seed = seed
        self.is_main_process = is_main_process

        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        self.metrics = build_metrics(metric_names if metric_names is not None else task.default_metric_names)

        self.epoch = 0
        self.global_step = 0

        self.model.to(device)

    def train_one_epoch(self) -> dict[str, float]:
        """Run one training epoch.

        Note (documented simplification, not a bug): if the number of
        batches in an epoch isn't a multiple of grad_accum_steps, the final
        partial accumulation window's gradients are discarded (never
        flushed) at epoch end — standard practice in most accumulation-loop
        implementations. With `build_dataloader`'s `drop_last=True` for the
        train split (Phase 3), this only ever affects the accumulation
        boundary, not individual sample loss.
        """
        import torch

        self.model.train()
        self.optimizer.zero_grad()

        total_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(self.train_loader):
            batch = _move_batch_to_device(batch, self.device)

            with self.precision_manager.autocast():
                result = compute_step(self.model, batch, self.task, self.loss_fn)
                scaled_loss = result["loss"] / self.grad_accum_steps

            self.precision_manager.backward(scaled_loss)

            is_accum_boundary = (i + 1) % self.grad_accum_steps == 0
            if is_accum_boundary:
                if self.grad_clip_norm is not None:
                    self.precision_manager.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

                self.precision_manager.step(self.optimizer)
                self.optimizer.zero_grad()

                if self.ema is not None:
                    self.ema.update(self.model)

                self.global_step += 1

                if self.is_main_process:
                    self.callbacks.on_batch_end(
                        {
                            "epoch": self.epoch,
                            "global_step": self.global_step,
                            "loss": float(result["loss"].item()),
                            "lr": self.optimizer.param_groups[0]["lr"],
                        }
                    )

            total_loss += float(result["loss"].item())
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return {"loss": total_loss / max(n_batches, 1)}

    def validate(self) -> dict[str, float]:
        import torch

        if self.val_loader is None:
            return {}

        self.model.eval()

        for m in self.metrics.values():
            m.reset()

        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = _move_batch_to_device(batch, self.device)

                result = compute_step(
                    self.model,
                    batch,
                    self.task,
                    loss_fn=self.loss_fn,
                )

                pred = result["prediction"]
                gt = batch["gt"]

                total_loss += float(result["loss"].item())
                n_batches += 1

                for b in range(pred.shape[0]):
                    for m in self.metrics.values():
                        m.update(pred[b, 0], gt[b, 0])

        metrics = {
            name: m.compute()
            for name, m in self.metrics.items()
        }

        metrics["loss"] = total_loss / max(n_batches, 1)

        return metrics

    def fit(self, num_epochs: int) -> None:
        start_state = self._base_state()
        
        if self.is_main_process:
            self.callbacks.on_train_start(start_state)

        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch

            if self.is_main_process:
                self.callbacks.on_epoch_start(self._base_state())

                
                # These MUST be outside the is_main_process block.
            train_metrics = self.train_one_epoch()
            val_metrics = self.validate()

            end_state = self._base_state()

            end_state["train_metrics"] = train_metrics
            end_state["val_metrics"] = val_metrics
            end_state["num_epochs"] = num_epochs
            end_state["val_loader"] = self.val_loader
            end_state["task"] = self.task
            end_state["device"] = self.device

            if self.is_main_process:
                self.callbacks.on_epoch_end(end_state)

            if end_state.get("stop_training", False):
                print(
                    f"\nTraining stopped early at epoch "
                    f"{self.epoch + 1}/{num_epochs}."
                )
                break

        if self.is_main_process:
            self.callbacks.on_train_end(self._base_state())

    def _base_state(self) -> dict[str, Any]:
        """Shared callback state dict — kept as a method so every hook call
        sees a consistent snapshot without duplicating this dict literal."""
        return {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "precision_manager": self.precision_manager,
            "ema": self.ema,
            "config": self.config,
            "seed": self.seed,
            "device": self.device,
        }

    def save_checkpoint(self, path: str) -> None:
        if not self.is_main_process:
            return  # avoid every DDP rank writing the same file simultaneously
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.precision_manager,
            ema=self.ema,
            epoch=self.epoch,
            global_step=self.global_step,
            config=self.config,
            seed=self.seed,
        )

    def load_checkpoint(self, path: str) -> None:
        state = load_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.precision_manager,
            ema=self.ema,
            map_location=self.device,
        )
        self.epoch = state["epoch"] + 1  # resume from the epoch AFTER the saved one
        self.global_step = state["global_step"]
