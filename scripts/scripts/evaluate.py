#!/usr/bin/env python3
"""Evaluate a trained checkpoint: full-split metrics + per-image CSV +
Markdown report + worst-sample visualizations (Phase 8's Evaluator).

Usage:
    python scripts/evaluate.py --model nafnet --checkpoint experiments/nafnet/run_001/best.pt \\
        --data-root dataset --split val --device cuda

Needs torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, default="restoration_sr")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--metrics", nargs="+", default=["psnr", "ssim", "lpips"])
    parser.add_argument("--primary-metric", type=str, default="psnr")
    parser.add_argument("--k-worst", type=int, default=10)
    parser.add_argument("--k-best", type=int, default=10)
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: alongside the checkpoint)")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("FATAL: torch is required for evaluation.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import tasks.denoising.task  # noqa: F401
    import tasks.restoration.task  # noqa: F401
    import tasks.super_resolution.task  # noqa: F401
    import models.restormer.model  # noqa: F401
    import metrics.psnr_metric  # noqa: F401
    import metrics.ssim_metric  # noqa: F401

    from datasets.restoration_dataset import build_dataloader, DatasetContractError
    from framework.evaluator.evaluator import Evaluator
    from framework.registry import MODEL_REGISTRY, TASK_REGISTRY
    from framework.trainer.checkpoint import load_checkpoint
    from metrics import build_metrics

    task = TASK_REGISTRY.build({"name": args.task})
    model = MODEL_REGISTRY.build({"name": args.model})
    task.check_model_compatibility(model.scale_factor)
    model.to(device)

    if not args.checkpoint.exists():
        print(f"FATAL: checkpoint not found at {args.checkpoint}")
        return 1
    load_checkpoint(args.checkpoint, model, map_location=device)
    print(f"Loaded checkpoint: {args.checkpoint}")

    try:
        loader = build_dataloader(args.data_root, args.split, batch_size=args.batch_size, strict=True)
    except DatasetContractError as exc:
        print(f"FATAL: dataset at {args.data_root} is not valid:\n{exc}")
        print("\nRun `python inspect_dataset.py` for the full report.")
        return 1

    out_dir = args.out or (args.checkpoint.parent / f"eval_report_{args.split}")

    evaluator = Evaluator(task, model, build_metrics(args.metrics), loader, device=device)
    print(f"Evaluating on '{args.split}' split ({len(loader.dataset)} samples)...")
    aggregate = evaluator.save_report(
        out_dir,
        primary_metric=args.primary_metric,
        k_worst=args.k_worst,
        k_best=args.k_best,
        split_name=args.split,
    )

    print(f"\nAggregate metrics: {aggregate}")
    print(f"Full report written to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
