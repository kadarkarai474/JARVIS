"""PyTorch Dataset for the AI Restoration Framework.

Built on top of two pieces validated in earlier phases:
    - utils.dataset_validator.validate_strict_pairing  (Phase 2 / 2.1 contract)
    - utils.data_transforms                              (normalization, crop, augment)

This module requires torch. It is intentionally a thin wrapper — all the
actual array logic lives in utils.data_transforms so it stays testable
without a torch install (see tests/test_data_transforms.py).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from utils.data_transforms import (
    VALID_NORMALIZATION_MODES,
    augment_pair,
    normalize,
    random_crop_pair,
)
from utils.dataset_validator import inspect_split, validate_strict_pairing


class DatasetContractError(RuntimeError):
    """Raised when a dataset split violates the strict shape/pairing contract."""


class RestorationDataset(Dataset):
    """Loads (NoisyLR, GT) pairs — or NoisyLR-only for the test split.

    Args:
        root: Path to the dataset root (containing train/val/test subfolders).
        split: One of "train", "val", "test".
        patch_size: If set, randomly crop a (patch_size, patch_size) NoisyLR
            patch (and matching 2x GT patch) per __getitem__ call. If None,
            returns the full 128x128 / 256x256 pair. Only applied for
            split="train" — val/test always return full images for
            comparable, reproducible evaluation.
        augment: Apply random flip/rotation. Only applied for split="train".
        normalization: One of utils.data_transforms.VALID_NORMALIZATION_MODES.
        normalization_stats: Required if normalization="dataset_stats", e.g.
            {"min": ..., "max": ...} taken from the Phase 2 validator report.
        seed: Base seed for this dataset's augmentation/crop RNG.
        strict: If True (default), raise DatasetContractError immediately if
            the split violates the strict pairing/shape rule instead of
            silently training on bad data.

    Raises:
        DatasetContractError: If strict=True and validation finds violations.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        patch_size: int | None = None,
        augment: bool = True,
        normalization: str = "clip_unit",
        normalization_stats: dict | None = None,
        seed: int = 0,
        strict: bool = True,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be train/val/test, got {split!r}")
        if normalization not in VALID_NORMALIZATION_MODES:
            raise ValueError(f"Unknown normalization mode: {normalization!r}")

        self.root = Path(root)
        self.split = split
        self.has_gt = split != "test"
        self.patch_size = patch_size if split == "train" else None
        self.augment = augment and split == "train"
        self.normalization = normalization
        self.normalization_stats = normalization_stats
        self.rng = np.random.default_rng(seed)

        noisy_dir = self.root / split / "NoisyLR"
        gt_dir = self.root / split / "GT" if self.has_gt else None

        split_report = inspect_split(split, noisy_dir, gt_dir)
        violations = validate_strict_pairing(split_report, require_gt=self.has_gt)
        if strict and violations:
            detail = "\n".join(f"  [{v.kind}] {v.detail}" for v in violations)
            raise DatasetContractError(
                f"Split '{split}' at {self.root} violates the strict dataset contract "
                f"({len(violations)} violation(s)). Run `python inspect_dataset.py` for the "
                f"full report. Violations:\n{detail}"
            )

        if self.has_gt:
            # Exclude anything flagged by strict validation (shape/pairing) AND
            # anything that failed to load at all — validate_strict_pairing only
            # shape-checks files that already loaded successfully, so corrupted
            # files must be excluded separately here or __getitem__ would crash.
            corrupted_stems = {f.path.stem for f in split_report.noisy_files if not f.loadable}
            corrupted_stems |= {f.path.stem for f in split_report.gt_files if not f.loadable}
            bad_stems = {v.stem for v in violations} | corrupted_stems
            self.stems = [s for s in split_report.matched_pairs if s not in bad_stems]
            self.noisy_dir = noisy_dir
            self.gt_dir = gt_dir
        else:
            corrupted_stems = {f.path.stem for f in split_report.noisy_files if not f.loadable}
            bad_stems = {v.stem for v in violations} | corrupted_stems
            self.stems = [
                f.path.stem for f in split_report.noisy_files if f.path.stem not in bad_stems
            ]
            self.noisy_dir = noisy_dir
            self.gt_dir = None

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        noisy = np.load(self.noisy_dir / f"{stem}.npy").astype(np.float32)

        if self.has_gt:
            gt = np.load(self.gt_dir / f"{stem}.npy").astype(np.float32)
        else:
            gt = None

        if self.patch_size is not None and gt is not None:
            noisy, gt = random_crop_pair(noisy, gt, self.patch_size, self.rng)

        if self.augment and gt is not None:
            noisy, gt = augment_pair(noisy, gt, self.rng)

        noisy = normalize(noisy, self.normalization, self.normalization_stats)
        noisy_t = torch.from_numpy(noisy).unsqueeze(0)  # (1, H, W)

        sample = {"noisy": noisy_t, "stem": stem}

        if gt is not None:
            gt = normalize(gt, self.normalization, self.normalization_stats)
            sample["gt"] = torch.from_numpy(gt).unsqueeze(0)  # (1, 2H, 2W)

        return sample


def build_dataloader(
    root: str | Path,
    split: str,
    batch_size: int = 1,
    num_workers: int = 0,
    patch_size: int | None = None,
    augment: bool = True,
    normalization: str = "clip_unit",
    normalization_stats: dict | None = None,
    strict: bool = True,
    seed: int = 0,
) -> DataLoader:
    """Convenience builder matching the RTX3050-dev-first strategy.

    Defaults to batch_size=1, num_workers=0 (single-process) — the "prove
    correctness first" starting point from the project's training-system
    plan. Increase both once the sanity model (Phase 11) confirms the
    pipeline works end-to-end.
    """
    dataset = RestorationDataset(
        root=root,
        split=split,
        patch_size=patch_size,
        augment=augment,
        normalization=normalization,
        normalization_stats=normalization_stats,
        seed=seed,
        strict=strict,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
