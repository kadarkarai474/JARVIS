#!/usr/bin/env python3
"""Train any registered model on any registered task.

Usage:
    python scripts/train.py --model sanity_cnn --task restoration_sr \\
        --data-root dataset --epochs 50 --device cuda --precision fp32

    python scripts/train.py --model nafnet --loss composite \\
        --loss-config configs/loss/l1_ssim_composite.yaml \\
        --data-root dataset --epochs 100 --device cuda --precision fp16 \\
        --grad-accum-steps 8 --use-ema

Every component (model/task/loss/optimizer/scheduler) is resolved through
its registry — this script contains no model-specific or loss-specific
logic, matching the project's "interchangeable via config" requirement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_yaml_or_name(value: str, default_params: dict | None = None) -> dict:
    """Accept either a bare registry name ("l1") or a path to a YAML config file."""
    import yaml

    path = Path(value)
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f)
    return {"name": value, "params": default_params or {}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a registered model.")
    parser.add_argument("--model", type=str, required=True, help="Registry name, e.g. 'nafnet'")
    parser.add_argument("--task", type=str, default="restoration_sr")
    parser.add_argument("--loss", type=str, default="l1", help="Registry name or path to a loss config YAML")
    parser.add_argument("--optimizer", type=str, default="adamw")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--scheduler", type=str, default=None, help="Registry name, e.g. 'cosine' (default: none)")
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim","lpips","mse"])
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint path to resume from")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("FATAL: torch is required to train. See requirements.txt / environment.yml.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Trigger registrations.
    import tasks.denoising.task  # noqa: F401
    import tasks.restoration.task  # noqa: F401
    import tasks.super_resolution.task  # noqa: F401
    import models.restormer.model  # noqa: F401
    import losses.l1_loss  # noqa: F401
    import losses.charbonnier_loss  # noqa: F401
    import losses.ssim_loss  # noqa: F401
    import losses.frequency_loss  # noqa: F401
    import losses.composite_loss  # noqa: F401
    import metrics.psnr_metric  # noqa: F401
    import metrics.ssim_metric  # noqa: F401
    import metrics.lpips_metric  # noqa: F401
    import metrics.mse_metric   # noqa: F401

    from datasets.restoration_dataset import build_dataloader, DatasetContractError
    from framework.callbacks.base import CallbackList
    from framework.callbacks.checkpoint_callback import CheckpointCallback
    from framework.callbacks.training_monitor_callback import (
        TrainingMonitorCallback,
    )
    from framework.callbacks.memory_monitor_callback import MemoryMonitorCallback
    from framework.registry import LOSS_REGISTRY, MODEL_REGISTRY, SCHEDULER_REGISTRY, TASK_REGISTRY
    from framework.trainer.optim import build_optimizer
    from framework.trainer.scheduler import build_scheduler
    from framework.trainer.trainer import Trainer
    from utils.experiment_dir import get_next_run_dir

    task = TASK_REGISTRY.build({"name": args.task})
    model = MODEL_REGISTRY.build({"name": args.model})
    task.check_model_compatibility(model.scale_factor)

    loss_cfg = _load_yaml_or_name(args.loss)
    loss_fn = LOSS_REGISTRY.build(loss_cfg)

    optimizer = build_optimizer({"name": args.optimizer, "params": {"lr": args.lr}}, model.parameters())
    scheduler = build_scheduler({"name": args.scheduler} if args.scheduler else None, optimizer)

    try:
        train_loader = build_dataloader(
            args.data_root, "train", batch_size=args.batch_size, patch_size=args.patch_size, strict=True
        )
        val_loader = build_dataloader(args.data_root, "val", batch_size=args.batch_size, strict=True)
    except DatasetContractError as exc:
        print(f"FATAL: dataset at {args.data_root} is not valid:\n{exc}")
        print("\nRun `python inspect_dataset.py` for the full report.")
        return 1

    run_dir = get_next_run_dir(f"experiments/{args.model}")
    print(f"Run directory: {run_dir}")

    callback_list = [
        CheckpointCallback(
            save_dir=run_dir,
            monitor=args.metrics[0],
            mode="max",
        ),
        TrainingMonitorCallback(
            save_dir=run_dir,
            patience=args.early_stop_patience,
            monitor="val_loss",
            mode="min",
        ),
        MemoryMonitorCallback(),
    ]
    if args.tensorboard:
        from framework.callbacks.tensorboard_callback import TensorBoardCallback

        callback_list.append(TensorBoardCallback(log_dir=str(run_dir / "tensorboard")))

    trainer = Trainer(
        task=task, model=model, loss_fn=loss_fn, optimizer=optimizer,
        train_loader=train_loader, val_loader=val_loader, scheduler=scheduler,
        callbacks=CallbackList(callback_list), device=device, precision=args.precision,
        grad_accum_steps=args.grad_accum_steps, grad_clip_norm=args.grad_clip_norm,
        use_ema=args.use_ema, metric_names=args.metrics,
    )

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)
        print(f"Resumed from {args.resume} at epoch {trainer.epoch}")

    trainer.fit(num_epochs=args.epochs)
    print(f"\nTraining complete. Checkpoints and logs in: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
