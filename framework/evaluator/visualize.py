"""Visualization for the evaluation report: side-by-side NoisyLR / Prediction /
GT comparisons for the worst-performing samples. Pure matplotlib, same
approach as Phase 2's dataset validator plots (headless Agg backend).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison_grid(
    samples: list[dict],
    out_path: str | Path,
    title: str = "Worst samples: NoisyLR / Prediction / GT",
) -> None:
    """Save a grid of NoisyLR/Prediction/GT triplets.

    Args:
        samples: list of {"stem": str, "noisy": (H,W) array, "prediction": (H,W)
            array, "gt": (H,W) array, and optionally the metric value(s) used
            to select this sample}.
        out_path: where to save the PNG.
        title: figure suptitle.
    """
    if not samples:
        return

    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = axes[None, :]

    for i, sample in enumerate(samples):
        axes[i, 0].imshow(sample["noisy"], cmap="gray")
        axes[i, 0].set_title(f"{sample['stem']} — NoisyLR")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.clip(sample["prediction"], 0, 1), cmap="gray")
        axes[i, 1].set_title("Prediction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(sample["gt"], cmap="gray")
        axes[i, 2].set_title("GT")
        axes[i, 2].axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_difference_map(
    prediction: np.ndarray,
    gt: np.ndarray,
    out_path: str | Path,
    title: str = "Absolute difference (Prediction - GT)",
) -> None:
    """Save a single |prediction - gt| heatmap — highlights exactly where the
    model is failing, useful for judge-facing material (Phase 19) too."""
    diff = np.abs(prediction.astype(np.float64) - gt.astype(np.float64))
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(diff, cmap="inferno")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
