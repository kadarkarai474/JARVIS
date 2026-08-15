#!/usr/bin/env python3
"""Benchmark any registered model — params, FLOPs/MACs, model size, latency,
GPU memory, throughput. Reused unchanged across Phases 13/15/17
(NAFNet/Restormer/SwinIR) since it only depends on MODEL_REGISTRY + the
model config, never a specific architecture.

Usage:
    python scripts/benchmark_model.py --model nafnet --device cuda
    python scripts/benchmark_model.py --model sanity_cnn --device cpu --warmup-iters 3 --timed-iters 5

Requires torch (times a real forward pass) -- unlike most of this
project's tooling, this cannot run in the build sandbox. Run it in your
real environment; see research_log.md Phase 13 entry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark a registered model.")
    parser.add_argument("--model", type=str, required=True, help="Registry name, e.g. 'nafnet'")
    parser.add_argument("--task", type=str, default="restoration_sr")
    parser.add_argument("--input-shape", type=int, nargs=4, default=[1, 1, 128, 128])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--timed-iters", type=int, default=50)
    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("FATAL: torch is required to benchmark a real forward pass.")
        return 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Trigger registrations for every known task/model — harmless if the
    # requested one isn't among these (a clear RegistryError follows).
    import tasks.denoising.task  # noqa: F401
    import tasks.restoration.task  # noqa: F401
    import tasks.super_resolution.task  # noqa: F401
    import models.sanity_cnn.model  # noqa: F401
    import models.nafnet.model  # noqa: F401
    import models.restormer.model  # noqa: F401
    import models.swinir.model  # noqa: F401

    from framework.benchmark.benchmark import Benchmark
    from framework.benchmark.report import write_benchmark_json, write_benchmark_markdown
    from framework.registry import MODEL_REGISTRY, TASK_REGISTRY
    from utils.experiment_dir import get_next_run_dir

    task = TASK_REGISTRY.build({"name": args.task})
    model = MODEL_REGISTRY.build({"name": args.model})
    task.check_model_compatibility(model.scale_factor)

    print(f"Benchmarking '{args.model}' on task '{args.task}' (device={device})...")
    bench = Benchmark(
        task, model, input_shape=tuple(args.input_shape), device=device,
        warmup_iters=args.warmup_iters, timed_iters=args.timed_iters,
    )
    results = bench.run()

    run_dir = get_next_run_dir(f"experiments/{args.model}")
    write_benchmark_json(results, run_dir / "benchmark_results.json")
    write_benchmark_markdown(results, args.model, run_dir / "BENCHMARK.md")

    print(f"\nResults written to {run_dir}")
    for key, value in results.items():
        print(f"  {key}: {value}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
