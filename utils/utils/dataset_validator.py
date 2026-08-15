"""Dataset validation utilities for the AI Restoration Framework.

This module implements the checks required before any dataset is trusted
for training or evaluation:

- sample counting
- shape verification (NoisyLR expected 128x128, GT expected 256x256)
- missing-pair detection (NoisyLR without matching GT, or vice versa)
- corrupted .npy detection
- dtype verification (expected float32)
- min / max / mean / std computation
- NaN / Inf detection
- histogram generation
- random visualizations
- input-GT side-by-side comparisons

Kept deliberately dependency-light (numpy + matplotlib only) since this tool
must run before the rest of the framework (Hydra configs, model code, etc.)
is trusted.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display available in CI / WSL without X server
import matplotlib.pyplot as plt
import numpy as np

EXPECTED_NOISY_SHAPE = (128, 128)
EXPECTED_GT_SHAPE = (256, 256)
EXPECTED_DTYPE = np.float32


@dataclasses.dataclass
class FileReport:
    """Per-file validation result."""

    path: Path
    loadable: bool = True
    error: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    min_val: float | None = None
    max_val: float | None = None
    mean_val: float | None = None
    std_val: float | None = None
    has_nan: bool = False
    has_inf: bool = False
    shape_ok: bool = False
    dtype_ok: bool = False


@dataclasses.dataclass
class SplitReport:
    """Validation summary for one split (train / val / test)."""

    split: str
    noisy_dir: Path
    gt_dir: Path | None
    noisy_files: list[FileReport] = dataclasses.field(default_factory=list)
    gt_files: list[FileReport] = dataclasses.field(default_factory=list)
    missing_gt_for: list[str] = dataclasses.field(default_factory=list)
    missing_noisy_for: list[str] = dataclasses.field(default_factory=list)
    matched_pairs: list[str] = dataclasses.field(default_factory=list)


def _inspect_file(path: Path, expected_shape: tuple[int, int]) -> FileReport:
    report = FileReport(path=path)
    try:
        arr = np.load(path)
    except Exception as exc:  # noqa: BLE001 — we want to catch and record any load failure
        report.loadable = False
        report.error = f"{type(exc).__name__}: {exc}"
        return report

    report.shape = arr.shape
    report.dtype = str(arr.dtype)
    report.shape_ok = arr.shape == expected_shape
    report.dtype_ok = arr.dtype == EXPECTED_DTYPE

    # NaN/Inf checks must run before float stats, since nan/inf poison min/max/mean.
    report.has_nan = bool(np.isnan(arr).any())
    report.has_inf = bool(np.isinf(arr).any())

    finite = arr[np.isfinite(arr)]
    if finite.size > 0:
        report.min_val = float(finite.min())
        report.max_val = float(finite.max())
        report.mean_val = float(finite.mean())
        report.std_val = float(finite.std())

    return report


def inspect_split(split: str, noisy_dir: Path, gt_dir: Path | None) -> SplitReport:
    """Validate one dataset split.

    Args:
        split: Name of the split ("train", "val", "test").
        noisy_dir: Directory containing NoisyLR .npy files.
        gt_dir: Directory containing GT .npy files, or None if this split
            has no ground truth (e.g. "test").

    Returns:
        A populated SplitReport.
    """
    report = SplitReport(split=split, noisy_dir=noisy_dir, gt_dir=gt_dir)

    noisy_paths = sorted(noisy_dir.glob("*.npy")) if noisy_dir.exists() else []
    gt_paths = sorted(gt_dir.glob("*.npy")) if (gt_dir and gt_dir.exists()) else []

    noisy_stems = {p.stem for p in noisy_paths}
    gt_stems = {p.stem for p in gt_paths}

    report.noisy_files = [_inspect_file(p, EXPECTED_NOISY_SHAPE) for p in noisy_paths]
    report.gt_files = [_inspect_file(p, EXPECTED_GT_SHAPE) for p in gt_paths]

    if gt_dir is not None:
        report.missing_gt_for = sorted(noisy_stems - gt_stems)
        report.missing_noisy_for = sorted(gt_stems - noisy_stems)
        report.matched_pairs = sorted(noisy_stems & gt_stems)

    return report


@dataclasses.dataclass
class StrictViolation:
    """A single strict-validation rule failure."""

    kind: str  # "shape_noisy" | "shape_gt" | "missing_gt" | "missing_noisy"
    detail: str
    stem: str  # the file stem this violation applies to, for programmatic filtering


def validate_strict_pairing(split_report: SplitReport, require_gt: bool) -> list[StrictViolation]:
    """Enforce the hard dataset contract for this project.

    Rule (fixed, not configurable — this is the data contract the whole
    framework is built on):
        - Every NoisyLR array must be exactly (128, 128).
        - Every GT array must be exactly (256, 256).
        - Filenames must pair *exactly* by stem between NoisyLR/ and GT/
          (e.g. "0000.npy" in NoisyLR must have a "0000.npy" in GT — no
          fuzzy matching, no index-based fallback, no tolerance for
          off-by-one naming).

    Args:
        split_report: Result of inspect_split() for one split.
        require_gt: Whether this split is expected to have GT at all
            (True for train/val, False for test).

    Returns:
        List of StrictViolation — empty means the split fully satisfies
        the contract.
    """
    violations: list[StrictViolation] = []

    for f in split_report.noisy_files:
        if f.loadable and f.shape != EXPECTED_NOISY_SHAPE:
            violations.append(
                StrictViolation(
                    "shape_noisy",
                    f"{f.path.name}: got {f.shape}, expected {EXPECTED_NOISY_SHAPE}",
                    stem=f.path.stem,
                )
            )

    for f in split_report.gt_files:
        if f.loadable and f.shape != EXPECTED_GT_SHAPE:
            violations.append(
                StrictViolation(
                    "shape_gt",
                    f"{f.path.name}: got {f.shape}, expected {EXPECTED_GT_SHAPE}",
                    stem=f.path.stem,
                )
            )

    if require_gt:
        for stem in split_report.missing_gt_for:
            violations.append(
                StrictViolation(
                    "missing_gt", f"NoisyLR/{stem}.npy has no exact-match GT/{stem}.npy", stem=stem
                )
            )
        for stem in split_report.missing_noisy_for:
            violations.append(
                StrictViolation(
                    "missing_noisy", f"GT/{stem}.npy has no exact-match NoisyLR/{stem}.npy", stem=stem
                )
            )

    return violations


def aggregate_stats(files: list[FileReport]) -> dict:
    """Aggregate per-file stats into split-level summary stats."""
    valid = [f for f in files if f.loadable and f.min_val is not None]
    if not valid:
        return {}
    mins = np.array([f.min_val for f in valid])
    maxs = np.array([f.max_val for f in valid])
    means = np.array([f.mean_val for f in valid])
    stds = np.array([f.std_val for f in valid])
    return {
        "n_valid": len(valid),
        "global_min": float(mins.min()),
        "global_max": float(maxs.max()),
        "mean_of_means": float(means.mean()),
        "mean_of_stds": float(stds.mean()),
        "n_nan": sum(f.has_nan for f in files),
        "n_inf": sum(f.has_inf for f in files),
        "n_shape_mismatch": sum(1 for f in files if f.loadable and not f.shape_ok),
        "n_dtype_mismatch": sum(1 for f in files if f.loadable and not f.dtype_ok),
        "n_corrupted": sum(1 for f in files if not f.loadable),
    }


def plot_histogram(files: list[FileReport], title: str, out_path: Path) -> None:
    """Plot an aggregate pixel-value histogram across all valid files."""
    values = []
    for f in files:
        if not f.loadable:
            continue
        try:
            arr = np.load(f.path)
            values.append(arr[np.isfinite(arr)].ravel())
        except Exception:  # noqa: BLE001 — already logged as corrupted elsewhere
            continue
    if not values:
        return
    all_values = np.concatenate(values)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(all_values, bins=100, color="#3b6ea5")
    ax.set_title(title)
    ax.set_xlabel("Pixel value")
    ax.set_ylabel("Count")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_random_pair_comparisons(
    matched_pairs: list[str],
    noisy_dir: Path,
    gt_dir: Path,
    out_path: Path,
    n_samples: int = 4,
    seed: int = 0,
) -> None:
    """Save a grid of random NoisyLR vs GT comparisons for a split.

    If no GT exists (matched_pairs empty), this is a no-op — caller should
    use plot_random_singles instead.
    """
    if not matched_pairs:
        return
    rng = np.random.default_rng(seed)
    chosen = rng.choice(matched_pairs, size=min(n_samples, len(matched_pairs)), replace=False)

    fig, axes = plt.subplots(len(chosen), 2, figsize=(6, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    for i, stem in enumerate(chosen):
        noisy = np.load(noisy_dir / f"{stem}.npy")
        gt = np.load(gt_dir / f"{stem}.npy")
        axes[i, 0].imshow(noisy, cmap="gray")
        axes[i, 0].set_title(f"{stem} — NoisyLR {noisy.shape}")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(gt, cmap="gray")
        axes[i, 1].set_title(f"{stem} — GT {gt.shape}")
        axes[i, 1].axis("off")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_random_singles(
    files: list[FileReport],
    out_path: Path,
    title: str,
    n_samples: int = 4,
    seed: int = 0,
) -> None:
    """Save a grid of random single-image visualizations (used when no GT exists)."""
    valid = [f for f in files if f.loadable]
    if not valid:
        return
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid, size=min(n_samples, len(valid)), replace=False)  # type: ignore[arg-type]

    fig, axes = plt.subplots(1, len(chosen), figsize=(3 * len(chosen), 3))
    if len(chosen) == 1:
        axes = [axes]

    for ax, f in zip(axes, chosen):
        arr = np.load(f.path)
        ax.imshow(arr, cmap="gray")
        ax.set_title(f"{f.path.stem} {arr.shape}", fontsize=9)
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
