"""Pure-numpy data transforms for the restoration dataset.

Deliberately kept free of any torch dependency so this logic can be unit
tested in any environment (including this build sandbox, which currently
has no torch installed) — only `datasets/restoration_dataset.py` wraps
these in torch tensors.
"""

from __future__ import annotations

import numpy as np

SCALE_FACTOR = 2  # GT is always 2x the NoisyLR resolution, per project spec

VALID_NORMALIZATION_MODES = ("none", "clip_unit", "dataset_stats")


def normalize(arr: np.ndarray, mode: str, stats: dict | None = None) -> np.ndarray:
    """Normalize a raw NoisyLR/GT array.

    Args:
        arr: Raw float32 array as loaded from disk.
        mode:
            "none"          - passthrough, no transform (use once real dataset
                               stats are known and a model-side norm layer is preferred).
            "clip_unit"     - clip to [0, 1]. Provisional default: cheap, bounded,
                               but will silently clip any legitimate values > 1.
            "dataset_stats" - affine rescale to [0, 1] using precomputed global
                               min/max (`stats={"min": ..., "max": ...}`), e.g. from
                               the Phase 2 validator report. Preferred once the real
                               dataset's true range is known.
        stats: Required only for mode="dataset_stats".

    Returns:
        Normalized float32 array.
    """
    if mode not in VALID_NORMALIZATION_MODES:
        raise ValueError(f"Unknown normalization mode: {mode!r}. Valid: {VALID_NORMALIZATION_MODES}")

    if mode == "none":
        return arr.astype(np.float32)

    if mode == "clip_unit":
        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    if mode == "dataset_stats":
        if not stats or "min" not in stats or "max" not in stats:
            raise ValueError('mode="dataset_stats" requires stats={"min": ..., "max": ...}')
        lo, hi = float(stats["min"]), float(stats["max"])
        if hi <= lo:
            raise ValueError(f"Invalid stats range: min={lo}, max={hi}")
        return ((arr - lo) / (hi - lo)).astype(np.float32)

    raise AssertionError("unreachable")  # pragma: no cover


def random_crop_pair(
    noisy: np.ndarray,
    gt: np.ndarray,
    patch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a matching (noisy, gt) patch pair at a random aligned location.

    The GT crop is always exactly SCALE_FACTOR times the NoisyLR crop, taken
    from the corresponding upscaled region, preserving pixel alignment
    between input and target.

    Args:
        noisy: (H, W) NoisyLR array, H == W == 128 in this project.
        gt: (2H, 2W) GT array.
        patch_size: Crop size for the NoisyLR side (must be <= noisy.shape[0]).
        rng: NumPy Generator for reproducible sampling.

    Returns:
        (noisy_patch, gt_patch) with shapes (patch_size, patch_size) and
        (patch_size * SCALE_FACTOR, patch_size * SCALE_FACTOR).
    """
    h, w = noisy.shape
    if patch_size > h or patch_size > w:
        raise ValueError(f"patch_size={patch_size} exceeds NoisyLR shape {noisy.shape}")
    if gt.shape != (h * SCALE_FACTOR, w * SCALE_FACTOR):
        raise ValueError(f"GT shape {gt.shape} does not match {SCALE_FACTOR}x NoisyLR shape {noisy.shape}")

    max_y = h - patch_size
    max_x = w - patch_size
    y = int(rng.integers(0, max_y + 1))
    x = int(rng.integers(0, max_x + 1))

    noisy_patch = noisy[y : y + patch_size, x : x + patch_size]
    gt_patch = gt[
        y * SCALE_FACTOR : (y + patch_size) * SCALE_FACTOR,
        x * SCALE_FACTOR : (x + patch_size) * SCALE_FACTOR,
    ]
    return noisy_patch, gt_patch


def augment_pair(
    noisy: np.ndarray,
    gt: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply an identical random geometric augmentation to a (noisy, gt) pair.

    Only flips and 90-degree rotations are used — these preserve pixel value
    statistics exactly (no interpolation), so they don't distort the noise
    characteristics the model needs to learn to remove.

    Args:
        noisy: (H, W) array.
        gt: (2H, 2W) array.
        rng: NumPy Generator.

    Returns:
        (noisy_aug, gt_aug), same shapes as input.
    """
    if rng.random() < 0.5:
        noisy = np.fliplr(noisy)
        gt = np.fliplr(gt)
    if rng.random() < 0.5:
        noisy = np.flipud(noisy)
        gt = np.flipud(gt)
    k = int(rng.integers(0, 4))
    if k:
        noisy = np.rot90(noisy, k)
        gt = np.rot90(gt, k)
    return np.ascontiguousarray(noisy), np.ascontiguousarray(gt)
