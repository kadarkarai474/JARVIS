"""Benchmark result reporting: JSON persistence + Markdown tables.

Pure stdlib (json module) + plain dicts — no torch dependency, so this is
testable now and reusable unchanged across every model benchmarked
(NAFNet here in Phase 13, Restormer/SwinIR in Phases 15/17, and the final
multi-model comparison in Phase 18).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Preferred display order + labels for known benchmark fields. Any fields
# not listed here still get included (appended, in whatever order dict
# iteration gives), so a new metric never silently disappears from the table.
_FIELD_LABELS: dict[str, str] = {
    "params": "Parameters",
    "model_size_mb": "Model Size (MB)",
    "flops": "FLOPs",
    "macs": "MACs",
    "mean_latency_ms": "Mean Latency (ms)",
    "std_latency_ms": "Latency Std (ms)",
    "throughput_images_per_sec": "Throughput (img/s)",
    "peak_allocated_mb": "Peak GPU Memory Allocated (MB)",
    "peak_reserved_mb": "Peak GPU Memory Reserved (MB)",
    "training_time_per_epoch_seconds": "Training Time/Epoch (s)",
    "psnr": "PSNR (dB)",
    "ssim": "SSIM",
    "lpips": "LPIPS",
}


def _format_value(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.4f}" if abs(value) < 1000 else f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def write_benchmark_json(results: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str))


def load_benchmark_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def write_benchmark_markdown(results: dict[str, Any], model_name: str, path: str | Path) -> None:
    """Write a single-model benchmark table."""
    lines = [f"# Benchmark: `{model_name}`", "", "| Metric | Value |", "|---|---|"]
    ordered_keys = [k for k in _FIELD_LABELS if k in results] + [
        k for k in results if k not in _FIELD_LABELS
    ]
    for key in ordered_keys:
        label = _FIELD_LABELS.get(key, key)
        lines.append(f"| {label} | {_format_value(results[key])} |")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_comparison_markdown(results_by_model: dict[str, dict[str, Any]], path: str | Path) -> None:
    """Write a multi-model comparison table (rows=models, cols=metrics) —
    used for Phase 18's final comparison, but built here so it's proven
    correct on the very first (single-model) benchmark run."""
    if not results_by_model:
        raise ValueError("results_by_model must contain at least one model")

    all_keys: list[str] = []
    for results in results_by_model.values():
        for key in results:
            if key not in all_keys:
                all_keys.append(key)
    ordered_keys = [k for k in _FIELD_LABELS if k in all_keys] + [
        k for k in all_keys if k not in _FIELD_LABELS
    ]

    header = ["Model"] + [_FIELD_LABELS.get(k, k) for k in ordered_keys]
    lines = ["# Model Comparison", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for model_name, results in results_by_model.items():
        row = [model_name] + [_format_value(results.get(k)) for k in ordered_keys]
        lines.append("| " + " | ".join(row) + " |")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
