#!/usr/bin/env python3
"""Short demo script for judges: loads a trained checkpoint, restores a few
sample images, and prints a compact before/after summary + saves a visual
comparison image.

Usage:
    python scripts/demo.py --model nafnet --checkpoint experiments/nafnet/run_001/best.pt \\
        --data-root dataset --device cuda --num-samples 4

Needs torch. Designed to run in under a minute for a live demo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Quick before/after demo for judges.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, default="restoration_sr")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/demo_output.png"))
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("FATAL: torch is required for the demo.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import tasks.restoration.task  # noqa: F401
    import models.sanity_cnn.model  # noqa: F401
    import models.nafnet.model  # noqa: F401
    import models.restormer.model  # noqa: F401
    import models.swinir.model  # noqa: F401
    import metrics.psnr_metric  # noqa: F401

    from datasets.restoration_dataset import build_dataloader
    from framework.evaluator.visualize import plot_comparison_grid
    from framework.registry import MODEL_REGISTRY, TASK_REGISTRY
    from framework.trainer.checkpoint import load_checkpoint
    from framework.trainer.step import compute_step
    from metrics.psnr_metric import PSNRMetric

    print(f"Loading '{args.model}' from {args.checkpoint} (device={device})...")
    task = TASK_REGISTRY.build({"name": args.task})
    model = MODEL_REGISTRY.build({"name": args.model})
    model.to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval()

    loader = build_dataloader(args.data_root, args.split, batch_size=1, strict=True)

    samples = []
    psnr_metric = PSNRMetric()
    print(f"\nRestoring {args.num_samples} sample(s) from the '{args.split}' split:\n")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= args.num_samples:
                break
            batch_dev = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            result = compute_step(model, batch_dev, task, loss_fn=None)
            pred = result["prediction"][0, 0]
            gt = batch_dev["gt"][0, 0]
            noisy = batch["noisy"][0, 0]
            stem = batch["stem"][0]

            psnr_metric.reset()
            psnr_metric.update(pred, gt)
            psnr_value = psnr_metric.compute()

            print(f"  [{stem}] PSNR = {psnr_value:.2f} dB")
            samples.append(
                {
                    "stem": stem,
                    "noisy": noisy.cpu().numpy(),
                    "prediction": pred.cpu().numpy(),
                    "gt": gt.cpu().numpy(),
                }
            )

    plot_comparison_grid(samples, args.out, title=f"{args.model} — before/after demo")
    print(f"\nComparison image saved to: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
