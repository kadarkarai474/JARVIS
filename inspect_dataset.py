#!/usr/bin/env python3
"""Dataset validator — the required first executable step of this project.

Usage:
    python inspect_dataset.py [--root dataset] [--out docs/dataset_report]

Produces a Markdown report plus supporting images covering, for every split
(train / val / test):
    - sample count
    - shape verification
    - missing-pair detection
    - corrupted .npy detection
    - dtype verification
    - min/max, mean/std
    - NaN / Inf detection
    - histograms
    - random visualizations
    - input-GT comparisons (where GT exists)

No training or evaluation code in this project should be trusted until this
script runs clean (or all reported issues are understood and accepted).
"""

from __future__ import annotations

import argparse
import datetime
import platform
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from utils.dataset_validator import (  # noqa: E402
    aggregate_stats,
    inspect_split,
    plot_histogram,
    plot_random_pair_comparisons,
    plot_random_singles,
    validate_strict_pairing,
)

SPLITS = {
    "train": {"noisy": "NoisyLR", "gt": "GT"},
    "val": {"noisy": "NoisyLR", "gt": "GT"},
    "test": {"noisy": "NoisyLR", "gt": None},  # test has no GT by design
}


def _fmt(x, digits=4):
    return "n/a" if x is None else f"{x:.{digits}f}"


def build_report(root: Path, out_dir: Path) -> tuple[str, dict[str, list]]:
    lines: list[str] = []
    all_violations: dict[str, list] = {}
    lines.append("# Dataset Validation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Dataset root: `{root}`")
    lines.append(f"- Python: {platform.python_version()}  |  NumPy: {np.__version__}")
    lines.append("")

    any_data_found = False
    any_corrupted = False
    any_nan_inf = False
    any_shape_mismatch = False

    for split, dirs in SPLITS.items():
        noisy_dir = root / split / dirs["noisy"]
        gt_dir = root / split / dirs["gt"] if dirs["gt"] else None

        split_report = inspect_split(split, noisy_dir, gt_dir)
        n_noisy = len(split_report.noisy_files)
        n_gt = len(split_report.gt_files)

        lines.append(f"## Split: `{split}`")
        lines.append("")
        lines.append(f"- NoisyLR dir: `{noisy_dir}` — {n_noisy} file(s) found")
        if gt_dir is not None:
            lines.append(f"- GT dir: `{gt_dir}` — {n_gt} file(s) found")
        else:
            lines.append("- GT dir: not applicable for this split (test set has no ground truth)")
        lines.append("")

        if n_noisy == 0 and n_gt == 0:
            lines.append("_No files found in this split — skipping detailed stats._")
            lines.append("")
            continue

        any_data_found = True

        # --- STRICT RULE: NoisyLR=(128,128), GT=(256,256), exact filename pairing ---
        require_gt = dirs["gt"] is not None
        violations = validate_strict_pairing(split_report, require_gt=require_gt)
        all_violations[split] = violations

        lines.append("### Strict validation rule")
        lines.append("")
        lines.append(
            "Rule: NoisyLR must be exactly `(128, 128)`, GT must be exactly `(256, 256)`, "
            "and every file must pair by **exact matching filename/stem** between "
            "`NoisyLR/` and `GT/` (no fuzzy or index-based matching)."
        )
        lines.append("")
        if violations:
            any_shape_mismatch = True
            lines.append(f"🔴 **{len(violations)} strict violation(s):**")
            lines.append("")
            for v in violations:
                lines.append(f"- `[{v.kind}]` {v.detail}")
            lines.append("")
        else:
            lines.append("🟢 No strict violations — all shapes and filename pairs are exact.")
            lines.append("")

        # --- Pairing ---
        if gt_dir is not None:
            lines.append(f"- Matched NoisyLR/GT pairs: **{len(split_report.matched_pairs)}**")
            if split_report.missing_gt_for:
                any_shape_mismatch = True  # flagged for the overall warning banner too
                lines.append(
                    f"- ⚠️ NoisyLR files with **no matching GT** ({len(split_report.missing_gt_for)}): "
                    f"{', '.join(split_report.missing_gt_for[:10])}"
                    + (" ..." if len(split_report.missing_gt_for) > 10 else "")
                )
            if split_report.missing_noisy_for:
                lines.append(
                    f"- ⚠️ GT files with **no matching NoisyLR** ({len(split_report.missing_noisy_for)}): "
                    f"{', '.join(split_report.missing_noisy_for[:10])}"
                    + (" ..." if len(split_report.missing_noisy_for) > 10 else "")
                )
            lines.append("")

        # --- Per-group stats table (NoisyLR, GT) ---
        for group_name, files, expected_shape in (
            ("NoisyLR", split_report.noisy_files, "(128, 128)"),
            ("GT", split_report.gt_files, "(256, 256)"),
        ):
            if not files:
                continue
            stats = aggregate_stats(files)
            corrupted = [f for f in files if not f.loadable]
            shape_bad = [f for f in files if f.loadable and not f.shape_ok]
            dtype_bad = [f for f in files if f.loadable and not f.dtype_ok]

            if corrupted:
                any_corrupted = True
            if stats.get("n_nan", 0) or stats.get("n_inf", 0):
                any_nan_inf = True
            if shape_bad:
                any_shape_mismatch = True

            lines.append(f"### `{split}` / {group_name} (expected shape {expected_shape}, dtype float32)")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|---|---|")
            lines.append(f"| Files found | {len(files)} |")
            lines.append(f"| Corrupted / unreadable | {len(corrupted)} |")
            lines.append(f"| Shape mismatches | {len(shape_bad)} |")
            lines.append(f"| Dtype mismatches | {len(dtype_bad)} |")
            lines.append(f"| Global min | {_fmt(stats.get('global_min'))} |")
            lines.append(f"| Global max | {_fmt(stats.get('global_max'))} |")
            lines.append(f"| Mean of per-file means | {_fmt(stats.get('mean_of_means'))} |")
            lines.append(f"| Mean of per-file stds | {_fmt(stats.get('mean_of_stds'))} |")
            lines.append(f"| Files containing NaN | {stats.get('n_nan', 0)} |")
            lines.append(f"| Files containing Inf | {stats.get('n_inf', 0)} |")
            lines.append("")

            if corrupted:
                lines.append("**Corrupted files:**")
                for f in corrupted:
                    lines.append(f"- `{f.path.name}`: {f.error}")
                lines.append("")

            if shape_bad:
                lines.append("**Shape mismatches:**")
                for f in shape_bad:
                    lines.append(f"- `{f.path.name}`: got {f.shape}, expected {expected_shape}")
                lines.append("")

            # Histogram
            hist_path = out_dir / "images" / f"{split}_{group_name.lower()}_histogram.png"
            plot_histogram(files, f"{split}/{group_name} pixel value distribution", hist_path)
            if hist_path.exists():
                rel = hist_path.relative_to(out_dir)
                lines.append(f"![{group_name} histogram]({rel})")
                lines.append("")

        # --- Visual comparisons ---
        if gt_dir is not None and split_report.matched_pairs:
            cmp_path = out_dir / "images" / f"{split}_input_gt_comparison.png"
            plot_random_pair_comparisons(
                split_report.matched_pairs, noisy_dir, gt_dir, cmp_path, n_samples=4
            )
            if cmp_path.exists():
                rel = cmp_path.relative_to(out_dir)
                lines.append(f"**Random NoisyLR vs GT comparisons:**")
                lines.append("")
                lines.append(f"![input-GT comparison]({rel})")
                lines.append("")
        elif split_report.noisy_files:
            single_path = out_dir / "images" / f"{split}_random_samples.png"
            plot_random_singles(
                split_report.noisy_files, single_path, f"{split} — random NoisyLR samples (no GT available)"
            )
            if single_path.exists():
                rel = single_path.relative_to(out_dir)
                lines.append("**Random samples (no GT available for comparison):**")
                lines.append("")
                lines.append(f"![random samples]({rel})")
                lines.append("")

    # --- Overall banner ---
    lines.insert(2, "")
    banner = []
    if not any_data_found:
        banner.append("> 🔴 **No data found in any split.** Nothing below is meaningful.")
    else:
        if any_corrupted:
            banner.append("> 🔴 **Corrupted files detected** — see per-split sections below.")
        if any_nan_inf:
            banner.append("> 🔴 **NaN/Inf values detected** — see per-split sections below.")
        if any_shape_mismatch:
            banner.append(
                "> 🟠 **Missing pairs or shape mismatches detected** — dataset is incomplete "
                "or inconsistent. Do not trust training results until resolved."
            )
        if not banner:
            banner.append("> 🟢 No corruption, NaN/Inf, or shape mismatches detected.")
    lines.insert(3, "\n".join(banner))
    lines.insert(4, "")

    return "\n".join(lines), all_violations


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the AI Restoration Framework dataset.")
    parser.add_argument("--root", type=Path, default=Path("dataset"), help="Path to dataset root")
    parser.add_argument(
        "--out", type=Path, default=Path("docs/dataset_report"), help="Output directory for the report"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Exit with a non-zero code if any strict shape/pairing violations are found (default: on).",
    )
    parser.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Report violations but always exit 0 (useful for exploratory runs).",
    )
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    report_md, all_violations = build_report(args.root, args.out)

    report_path = args.out / "REPORT.md"
    report_path.write_text(report_md, encoding="utf-8")

    print(f"Dataset report written to: {report_path}")
    print(f"Supporting images written to: {args.out / 'images'}")

    total_violations = sum(len(v) for v in all_violations.values())
    if total_violations:
        print(f"\n⚠️  {total_violations} strict violation(s) found:")
        for split, violations in all_violations.items():
            for v in violations:
                print(f"  [{split}] [{v.kind}] {v.detail}")
        if args.strict:
            print("\nExiting non-zero due to --strict (default). Use --no-strict to suppress.")
            sys.exit(1)
    else:
        print("\n✅ Strict validation passed: all shapes and filename pairs are exact.")


if __name__ == "__main__":
    main()
