"""Report writers for the evaluation system: per-image CSV, Markdown summary,
and difficult-sample identification. Pure stdlib (csv module) — no new
dependency for something this simple, and it's fully testable without torch
since it operates on plain dicts/lists of numbers.
"""

from __future__ import annotations

import csv
import datetime
from pathlib import Path
from typing import Any


def write_per_image_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    """Write per-image metric rows to CSV. `rows` is a list of dicts, each
    with a "stem" key plus one key per metric name (as produced by
    Evaluator.evaluate_per_image())."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("stem\n")  # empty split — still produce a valid (header-only) CSV
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_worst_samples(
    rows: list[dict[str, Any]], metric_name: str, k: int = 10, mode: str = "min"
) -> list[dict[str, Any]]:
    """Return the k worst-performing samples by `metric_name`.

    Args:
        mode: "min" if lower is worse (e.g. PSNR, SSIM — lower means worse
            quality), "max" if higher is worse (rare, but kept general).
    """
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
    valid_rows = [r for r in rows if r.get(metric_name) is not None]
    reverse = mode == "max"  # if higher is worse, sort descending to put worst first
    sorted_rows = sorted(valid_rows, key=lambda r: r[metric_name], reverse=reverse)
    return sorted_rows[:k]
def find_best_samples(
    rows: list[dict[str, Any]], metric_name: str, k: int = 10, mode: str = "min"
) -> list[dict[str, Any]]:
    """Return the k best-performing samples by metric_name.

    For PSNR/SSIM: higher is better.
    For LPIPS: lower is better.
    """
    if mode not in ("min", "max"):
        raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")

    valid_rows = [r for r in rows if r.get(metric_name) is not None]

    # If lower is worse (PSNR/SSIM), higher values are best.
    # If higher is worse (e.g. LPIPS if configured that way), lower values are best.
    reverse = mode == "min"

    sorted_rows = sorted(
        valid_rows,
        key=lambda r: r[metric_name],
        reverse=reverse,
    )

    return sorted_rows[:k]

def write_summary_markdown(
    aggregate_metrics: dict[str, float],
    per_image_rows: list[dict[str, Any]],
    worst_samples: list[dict[str, Any]],
    worst_metric_name: str,
    path: str | Path,
    split_name: str = "val",
) -> None:
    """Write a human-readable Markdown evaluation summary."""
    lines = [
        f"# Evaluation Report — split: `{split_name}`",
        "",
        f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}",
        f"- Samples evaluated: {len(per_image_rows)}",
        "",
        "## Dataset-average metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name, value in aggregate_metrics.items():
        lines.append(f"| {name} | {value:.4f} |")
    lines.append("")

    lines.append(f"## Worst {len(worst_samples)} samples by `{worst_metric_name}`")
    lines.append("")
    if worst_samples:
        cols = list(worst_samples[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "---|" * len(cols))
        for row in worst_samples:
            values = [f"{row[c]:.4f}" if isinstance(row[c], float) else str(row[c]) for c in cols]
            lines.append("| " + " | ".join(values) + " |")
    else:
        lines.append("_No samples available._")
    lines.append("")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
