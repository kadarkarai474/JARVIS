"""Experiment run directory helper.

Per the project's experiment-management rule: never overwrite a past run.
Each call to `get_next_run_dir` returns a fresh, not-yet-existing
`run_NNN` directory under the given base path (e.g.
`experiments/sanity_cnn/run_003`), regardless of gaps or non-sequential
existing run numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

_RUN_DIR_PATTERN = re.compile(r"^run_(\d+)$")


def get_next_run_dir(base_dir: str | Path, create: bool = True) -> Path:
    """Return the next unused `run_NNN` directory under `base_dir`.

    Args:
        base_dir: e.g. "experiments/sanity_cnn".
        create: If True (default), create the directory (and base_dir) now.
            If False, only compute the path — useful when the caller wants
            to decide the exact moment of creation.

    Returns:
        Path to the new run directory, e.g. "experiments/sanity_cnn/run_004".
    """
    base_dir = Path(base_dir)
    existing_numbers = []
    if base_dir.exists():
        for entry in base_dir.iterdir():
            if entry.is_dir():
                match = _RUN_DIR_PATTERN.match(entry.name)
                if match:
                    existing_numbers.append(int(match.group(1)))

    next_number = (max(existing_numbers) + 1) if existing_numbers else 1
    run_dir = base_dir / f"run_{next_number:03d}"

    if create:
        run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir
