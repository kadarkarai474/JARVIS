"""Latency statistics: pure arithmetic over a list of per-iteration timings.

Kept separate from benchmark.py (which needs torch to actually time a
forward pass) so the aggregation math — mean, std, min, max, throughput —
is independently testable without a GPU or torch install.
"""

from __future__ import annotations

import statistics


def compute_latency_stats(timings_seconds: list[float], batch_size: int = 1) -> dict[str, float]:
    """Aggregate a list of raw per-iteration wall-clock timings.

    Args:
        timings_seconds: One float per timed iteration (already excluding
            warmup iterations — those should never be passed in here).
        batch_size: Samples processed per timed iteration, for throughput.

    Returns:
        Dict with mean/std/min/max latency in milliseconds, plus throughput
        in images/sec (batch_size / mean_latency_seconds).
    """
    if not timings_seconds:
        raise ValueError("compute_latency_stats requires at least one timing")

    mean_s = statistics.mean(timings_seconds)
    std_s = statistics.stdev(timings_seconds) if len(timings_seconds) > 1 else 0.0

    return {
        "mean_latency_ms": mean_s * 1000,
        "std_latency_ms": std_s * 1000,
        "min_latency_ms": min(timings_seconds) * 1000,
        "max_latency_ms": max(timings_seconds) * 1000,
        "throughput_images_per_sec": batch_size / mean_s if mean_s > 0 else float("inf"),
    }


def compute_model_size_mb(param_numels_and_bytes: list[tuple[int, int]]) -> float:
    """Compute model size in MB from a list of (numel, element_size_bytes) pairs.

    Factored out from Benchmark.run() (which gets these values from
    `model.parameters()`) so the arithmetic is testable with plain tuples,
    no torch tensors required.
    """
    total_bytes = sum(numel * elem_size for numel, elem_size in param_numels_and_bytes)
    return total_bytes / (1024**2)
