#!/usr/bin/env python3
"""Run inference with a trained checkpoint on the GT-less test/ split (or any
NoisyLR-only folder), saving predictions as .npy (Phase 9's Inferencer).

Usage:
    python scripts/infer.py --model nafnet --checkpoint experiments/nafnet/run_001/best.pt \\
        --data-root dataset --device cuda

Needs torch.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run inference with a trained checkpoint.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, default="restoration_sr")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("dataset"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None, help="Output dir (default: alongside the checkpoint)")
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("FATAL: torch is required for inference.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    import tasks.denoising.task  # noqa: F401
    import tasks.restoration.task  # noqa: F401
    import tasks.super_resolution.task  # noqa: F401
    import models.restormer.model  # noqa: F401

    from datasets.restoration_dataset import build_dataloader, DatasetContractError
    from framework.inference.inferencer import Inferencer
    from framework.registry import MODEL_REGISTRY, TASK_REGISTRY
    from framework.trainer.checkpoint import load_checkpoint

    task = TASK_REGISTRY.build({"name": args.task})
    model = MODEL_REGISTRY.build({"name": args.model})
    task.check_model_compatibility(model.scale_factor)
    model.to(device)

    if not args.checkpoint.exists():
        print(f"FATAL: checkpoint not found at {args.checkpoint}")
        return 1
    import torch

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_state = state["model"]

    # Remove FLOPs/parameter profiling buffers from the checkpoint.
    profiling_keys = [
        key for key in model_state
        if key.endswith(".total_ops")
        or key.endswith(".total_params")
        or key in {"total_ops", "total_params"}
    ]

    for key in profiling_keys:
        del model_state[key]

    model.load_state_dict(model_state, strict=True)

    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Ignored profiling keys: {len(profiling_keys)}")

    try:
        loader = build_dataloader(args.data_root, args.split, batch_size=args.batch_size, strict=True)
    except DatasetContractError as exc:
        print(f"FATAL: dataset at {args.data_root} is not valid:\n{exc}")
        print("\nRun `python inspect_dataset.py` for the full report.")
        return 1

    out_dir = args.out or (args.checkpoint.parent / "predictions")

    inferencer = Inferencer(task, model, device=device, output_dir=out_dir)
    print(f"Running inference on '{args.split}' split ({len(loader.dataset)} samples)...")
    results = inferencer.run(loader)

    print(f"\n{len(results)} predictions saved to: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
