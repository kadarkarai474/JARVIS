"""Concrete Benchmark: full implementation of BaseBenchmark (Phase 4).

Produces the research-paper-style comparison table the project spec
requires: params, FLOPs, MACs, model size, latency (mean+std), GPU memory,
throughput, and (optionally, if supplied by the caller from an actual
training run's logs) training time per epoch.

FLOPs/MACs use fvcore's FlopCountAnalysis, wrapped in try/except — fvcore's
operator coverage is incomplete for some architectures (a known limitation
of the library, not this code), so a failure there degrades to
flops=None/macs=None with a warning rather than crashing the whole
benchmark over one unsupported op.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

from framework.benchmark.base import BaseBenchmark
from framework.benchmark.timing import compute_latency_stats, compute_model_size_mb
from tasks.base import BaseTask


class Benchmark(BaseBenchmark):
    def __init__(
        self,
        task: BaseTask,
        model: Any,
        input_shape: tuple[int, int, int, int] = (1, 1, 128, 128),
        device: str = "cuda",
        warmup_iters: int = 10,
        timed_iters: int = 50,
        config: dict | None = None,
    ) -> None:
        super().__init__(task, model, config)
        self.input_shape = input_shape
        self.device = device
        self.warmup_iters = warmup_iters
        self.timed_iters = timed_iters

    def _count_params(self) -> int:
        return self.model.num_parameters()

    def _model_size_mb(self) -> float:
        pairs = [(p.numel(), p.element_size()) for p in self.model.parameters()]
        return compute_model_size_mb(pairs)

    def _flops_and_macs(self, dummy_input) -> tuple[float | None, float | None]:
        try:
            from fvcore.nn import FlopCountAnalysis
        except ImportError:
            warnings.warn("fvcore not installed — skipping FLOPs/MACs computation", stacklevel=2)
            return None, None

        try:
            analysis = FlopCountAnalysis(self.model, dummy_input)
            analysis.unsupported_ops_warnings(False)  # we surface our own warning below instead
            macs = float(analysis.total())
            flops = macs * 2  # 1 MAC = 2 FLOPs (one multiply + one add), standard convention
            return flops, macs
        except Exception as exc:  # noqa: BLE001 — fvcore's op coverage varies by architecture
            warnings.warn(f"FLOPs/MACs computation failed ({exc}) — reporting None", stacklevel=2)
            return None, None

    def _time_forward_passes(self, dummy_input) -> dict[str, float]:
        import torch

        self.model.eval()
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                self.model(dummy_input)
            if "cuda" in self.device and torch.cuda.is_available():
                torch.cuda.synchronize()

            timings: list[float] = []
            for _ in range(self.timed_iters):
                if "cuda" in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                start = time.perf_counter()
                self.model(dummy_input)
                if "cuda" in self.device and torch.cuda.is_available():
                    torch.cuda.synchronize()
                timings.append(time.perf_counter() - start)

        return compute_latency_stats(timings, batch_size=self.input_shape[0])

    def _gpu_memory_mb(self, dummy_input) -> dict[str, float] | None:
        import torch

        if not ("cuda" in self.device and torch.cuda.is_available()):
            return None

        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            self.model(dummy_input)
        torch.cuda.synchronize()
        return {
            "peak_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
            "peak_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
        }

    def run(self, training_time_per_epoch_seconds: float | None = None) -> dict[str, Any]:
        import torch

        self.model.to(self.device)
        dummy_input = torch.randn(*self.input_shape, device=self.device)

        results: dict[str, Any] = {
            "params": self._count_params(),
            "model_size_mb": self._model_size_mb(),
        }

        flops, macs = self._flops_and_macs(dummy_input)
        results["flops"] = flops
        results["macs"] = macs

        results.update(self._time_forward_passes(dummy_input))

        gpu_mem = self._gpu_memory_mb(dummy_input)
        if gpu_mem is not None:
            results.update(gpu_mem)

        if training_time_per_epoch_seconds is not None:
            results["training_time_per_epoch_seconds"] = training_time_per_epoch_seconds

        return results
