"""Abstract evaluator interface. Full implementation: Phase 8.

Reuses framework.trainer.step.compute_step for the actual forward pass —
evaluation is just compute_step + metric accumulation over a full split,
with no gradient updates. This shared-step design is what lets Phase 8
avoid re-deriving forward-pass logic already proven in Phase 7.
"""

from __future__ import annotations

import abc
from typing import Any

from tasks.base import BaseTask


class BaseEvaluator(abc.ABC):
    """Abstract evaluator. Concrete implementation: Phase 8."""

    def __init__(
        self,
        task: BaseTask,
        model: Any,
        metrics: dict[str, Any],
        data_loader: Any,
        config: dict | None = None,
    ) -> None:
        self.task = task
        self.model = model
        self.metrics = metrics  # name -> BaseMetric instance, from METRIC_REGISTRY
        self.data_loader = data_loader
        self.config = config or {}

    @abc.abstractmethod
    def evaluate(self) -> dict[str, float]:
        """Run the full split, return {metric_name: aggregated_value}. Phase 8."""
        raise NotImplementedError

    @abc.abstractmethod
    def evaluate_per_image(self) -> list[dict[str, Any]]:
        """Return per-image metric rows for CSV export. Phase 8."""
        raise NotImplementedError
