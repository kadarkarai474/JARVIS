"""Abstract benchmark interface. Full implementation: Phase 10.

Will report params, FLOPs/MACs, model size, latency (mean+std), GPU
memory, throughput, and training time/epoch per the project's
research-paper-style benchmark table requirement — for any model, since
it only depends on BaseRestorationModel.num_parameters() and the shared
compute_step for timing forward passes.
"""

from __future__ import annotations

import abc
from typing import Any

from tasks.base import BaseTask


class BaseBenchmark(abc.ABC):
    """Abstract benchmark harness. Concrete implementation: Phase 10."""

    def __init__(self, task: BaseTask, model: Any, config: dict | None = None) -> None:
        self.task = task
        self.model = model
        self.config = config or {}

    @abc.abstractmethod
    def run(self) -> dict[str, Any]:
        """Return a dict of benchmark metrics (params, FLOPs, latency, memory, throughput...). Phase 10."""
        raise NotImplementedError
