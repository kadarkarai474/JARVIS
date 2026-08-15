"""Checkpoint save/load, plus the reproducibility metadata every run must record
per the project's experiment-tracking spec: config, seeds, Python/PyTorch/CUDA
versions, GPU info, package versions, git commit hash.
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def get_git_commit_hash(repo_dir: str | Path = ".") -> str:
    """Return the current git commit hash, or "unknown" if unavailable.

    Never raises — a missing git repo, detached environment, or git not
    being installed should degrade to "unknown" rather than crash a
    training run over a metadata nicety.
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir),
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def collect_env_info() -> dict[str, Any]:
    """Collect reproducibility-relevant environment info.

    torch is imported lazily inside this function (not at module level) so
    that get_git_commit_hash() and the rest of this module's non-torch
    logic remain usable/testable even without torch installed.
    """
    info: dict[str, Any] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": get_git_commit_hash(),
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False
    return info


def save_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    ema: Any = None,
    epoch: int = 0,
    global_step: int = 0,
    best_metric: float | None = None,
    config: dict | None = None,
    seed: int | None = None,
) -> None:
    """Save full trainer state for exact resume, plus reproducibility metadata."""
    import torch

    state: dict[str, Any] = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "config": config,
        "seed": seed,
        "env": collect_env_info(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(
    path: str | Path,
    model: Any,
    optimizer: Any = None,
    scheduler: Any = None,
    scaler: Any = None,
    ema: Any = None,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Restore trainer state from a checkpoint. Returns the full state dict
    (epoch/global_step/best_metric/config/env) so the caller can resume
    exactly where training left off.
    """
    import torch

    state = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(state["model"])
    if optimizer is not None and state.get("optimizer") is not None:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler") is not None:
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    return state
