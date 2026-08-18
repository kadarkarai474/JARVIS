#!/usr/bin/env python3

"""
JARVIS End-to-End AI Image Restoration Inference

Usage:
    python run.py <input_dir> <output_dir>

Example:
    python run.py /content/JARVIS_dataset/test/NoisyLR /content/JARVIS/output
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# ================================================================
# CONFIGURATION
# ================================================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_NAME = "restormer"
TASK_NAME = "restoration_sr"

CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "best.pt"
)

INFER_SCRIPT = PROJECT_ROOT / "scripts" / "infer.py"

DEFAULT_BATCH_SIZE = 8


# ================================================================
# INPUT VALIDATION
# ================================================================

def validate_input_directory(input_dir: Path) -> list[Path]:

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory does not exist: {input_dir}"
        )

    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"Input path is not a directory: {input_dir}"
        )

    files = sorted(input_dir.glob("*.npy"))

    if not files:
        raise RuntimeError(
            f"No .npy files found in input directory: {input_dir}"
        )

    return files


# ================================================================
# OUTPUT VALIDATION
# ================================================================

def validate_outputs(
    input_files: list[Path],
    output_dir: Path,
) -> bool:

    output_files = sorted(output_dir.glob("*.npy"))

    print()
    print("=" * 72)
    print("OUTPUT VALIDATION")
    print("=" * 72)

    print(f"Input files   : {len(input_files)}")
    print(f"Output files  : {len(output_files)}")
    print(f"Output folder : {output_dir}")

    expected = {p.name for p in input_files}
    actual = {p.name for p in output_files}

    missing = sorted(expected - actual)
    extra = sorted(actual - expected)

    if missing:
        print(f"Missing outputs: {len(missing)}")

        for name in missing[:10]:
            print(f"  {name}")

    if extra:
        print(f"Unexpected outputs: {len(extra)}")

        for name in extra[:10]:
            print(f"  {name}")

    if not missing and not extra:
        print("Output validation PASSED.")
        return True

    print("Output validation FAILED.")
    return False


# ================================================================
# DEVICE
# ================================================================

def get_device() -> str:

    try:
        import torch
    except ImportError:
        print("FATAL: PyTorch is not installed.")
        return "unavailable"

    return "cuda" if torch.cuda.is_available() else "cpu"


# ================================================================
# MAIN
# ================================================================

def main() -> int:

    # ------------------------------------------------------------
    # END-TO-END TIMER START
    # ------------------------------------------------------------

    total_start = time.perf_counter()

    # ------------------------------------------------------------
    # ARGUMENTS
    # ------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="JARVIS end-to-end Restormer inference."
    )

    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing NoisyLR .npy files.",
    )

    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where restored .npy files are saved.",
    )

    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    # ------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------

    print("=" * 72)
    print("JARVIS AI-BASED IMAGE RESTORATION")
    print("=" * 72)

    print(f"Project root : {PROJECT_ROOT}")
    print(f"Model        : {MODEL_NAME}")
    print(f"Task         : {TASK_NAME}")
    print(f"Checkpoint   : {CHECKPOINT}")
    print(f"Input        : {input_dir}")
    print(f"Output       : {output_dir}")

    # ------------------------------------------------------------
    # INPUT VALIDATION TIMER
    # ------------------------------------------------------------

    validation_start = time.perf_counter()

    try:
        input_files = validate_input_directory(input_dir)
    except Exception as exc:
        print(f"\nFATAL: {exc}")
        return 1

    validation_time = time.perf_counter() - validation_start

    print(f"Input files  : {len(input_files)}")

    # ------------------------------------------------------------
    # CHECKPOINT
    # ------------------------------------------------------------

    if not CHECKPOINT.is_file():

        print("\nFATAL: checkpoint not found.")
        print(f"Expected: {CHECKPOINT}")

        experiments_dir = PROJECT_ROOT / "experiments"

        if experiments_dir.exists():

            print("\nAvailable checkpoints:")

            found = list(
                experiments_dir.rglob("*.pt")
            )

            for checkpoint in found:
                print(f"  {checkpoint}")

        return 1

    # ------------------------------------------------------------
    # INFER SCRIPT
    # ------------------------------------------------------------

    if not INFER_SCRIPT.is_file():

        print("\nFATAL: inference script not found.")
        print(f"Expected: {INFER_SCRIPT}")

        return 1

    # ------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------

    device_start = time.perf_counter()

    device = get_device()

    if device == "unavailable":
        return 1

    device_time = time.perf_counter() - device_start

    print(f"Device       : {device}")

    if device == "cuda":

        import torch

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA         : "
            f"{torch.version.cuda}"
        )

    # ------------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # DATA PREPARATION + INFERENCE
    # ------------------------------------------------------------

    inference_start = time.perf_counter()

    with tempfile.TemporaryDirectory(
        prefix="jarvis_inference_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        test_dir = temp_root / "test"
        noisy_lr_dir = test_dir / "NoisyLR"

        test_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # --------------------------------------------------------
        # Try symbolic link first.
        # This avoids copying the dataset.
        # --------------------------------------------------------

        try:

            noisy_lr_dir.symlink_to(
                input_dir,
                target_is_directory=True,
            )

            print("Input mode   : symlink")

        except OSError:

            print(
                "Input mode   : copy "
                "(symlink unavailable)"
            )

            noisy_lr_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            for source_file in input_files:

                shutil.copy2(
                    source_file,
                    noisy_lr_dir / source_file.name,
                )

        # --------------------------------------------------------
        # INFERENCE COMMAND
        # --------------------------------------------------------

        command = [
            sys.executable,
            str(INFER_SCRIPT),

            "--model",
            MODEL_NAME,

            "--task",
            TASK_NAME,

            "--checkpoint",
            str(CHECKPOINT),

            "--data-root",
            str(temp_root),

            "--split",
            "test",

            "--batch-size",
            str(DEFAULT_BATCH_SIZE),

            "--device",
            device,

            "--out",
            str(output_dir),
        ]

        print()
        print("=" * 72)
        print("STARTING INFERENCE")
        print("=" * 72)

        print(f"Batch size   : {DEFAULT_BATCH_SIZE}")
        print(f"Samples      : {len(input_files)}")
        print()

        # --------------------------------------------------------
        # RUN INFERENCE
        # --------------------------------------------------------

        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
        )

        if result.returncode != 0:

            inference_time = (
                time.perf_counter()
                - inference_start
            )

            total_time = (
                time.perf_counter()
                - total_start
            )

            print()
            print("=" * 72)
            print("FATAL: INFERENCE FAILED")
            print("=" * 72)

            print(
                f"Process exit code : "
                f"{result.returncode}"
            )

            print(
                f"Inference time    : "
                f"{inference_time:.3f} s"
            )

            print(
                f"Total time        : "
                f"{total_time:.3f} s"
            )

            return result.returncode

    # ------------------------------------------------------------
    # INFERENCE TIMER END
    # ------------------------------------------------------------

    inference_time = (
        time.perf_counter()
        - inference_start
    )

    # ------------------------------------------------------------
    # OUTPUT VALIDATION
    # ------------------------------------------------------------

    validation_output_start = time.perf_counter()

    success = validate_outputs(
        input_files,
        output_dir,
    )

    output_validation_time = (
        time.perf_counter()
        - validation_output_start
    )

    if not success:
        return 1

    # ------------------------------------------------------------
    # END-TO-END TIMER END
    # ------------------------------------------------------------

    total_time = (
        time.perf_counter()
        - total_start
    )

    num_images = len(input_files)

    avg_time_ms = (
        total_time / num_images
    ) * 1000.0

    inference_avg_ms = (
        inference_time / num_images
    ) * 1000.0

    throughput = (
        num_images / inference_time
        if inference_time > 0
        else 0.0
    )

    end_to_end_throughput = (
        num_images / total_time
        if total_time > 0
        else 0.0
    )

    # ------------------------------------------------------------
    # FINAL BENCHMARK
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("END-TO-END INFERENCE BENCHMARK")
    print("=" * 72)

    print(
        f"Images                 : "
        f"{num_images}"
    )

    print(
        f"Batch size             : "
        f"{DEFAULT_BATCH_SIZE}"
    )

    print(
        f"Device                 : "
        f"{device}"
    )

    print(
        f"Input validation       : "
        f"{validation_time * 1000:.3f} ms"
    )

    print(
        f"Device detection       : "
        f"{device_time * 1000:.3f} ms"
    )

    print(
        f"Inference time         : "
        f"{inference_time:.3f} s"
    )

    print(
        f"Average inference      : "
        f"{inference_avg_ms:.3f} ms/image"
    )

    print(
        f"Inference throughput   : "
        f"{throughput:.2f} images/sec"
    )

    print(
        f"Output validation      : "
        f"{output_validation_time * 1000:.3f} ms"
    )

    print(
        f"TOTAL END-TO-END TIME  : "
        f"{total_time:.3f} s"
    )

    print(
        f"END-TO-END AVG         : "
        f"{avg_time_ms:.3f} ms/image"
    )

    print(
        f"END-TO-END THROUGHPUT  : "
        f"{end_to_end_throughput:.2f} images/sec"
    )

    print("=" * 72)

    print()
    print("INFERENCE COMPLETED SUCCESSFULLY")
    print(f"Predictions saved to: {output_dir}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

