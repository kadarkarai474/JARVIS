"""Abstract inference interface. Full implementation: Phase 9.

Also built on compute_step (with loss_fn=None), so a trained model can be
run on new NoisyLR inputs (no GT required, matching the test/ split
having no ground truth) without any task/model-specific inference code.
"""

from __future__ import annotations

import abc
from typing import Any

from tasks.base import BaseTask


class BaseInferencer(abc.ABC):
    """Abstract inferencer. Concrete implementation: Phase 9."""

    def __init__(self, task: BaseTask, model: Any, config: dict | None = None) -> None:
        self.task = task
        self.model = model
        self.config = config or {}

    @abc.abstractmethod
    def run(self, data_loader: Any) -> list[dict[str, Any]]:
        """Run inference over a NoisyLR-only data loader, return per-sample results. Phase 9."""
        raise NotImplementedError

    @abc.abstractmethod
    def run_single(self, noisy: Any) -> Any:
        """Run inference on one in-memory NoisyLR array/tensor. Phase 9."""
        raise NotImplementedError
