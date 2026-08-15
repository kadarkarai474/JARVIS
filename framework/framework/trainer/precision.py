"""Mixed-precision management: FP32 / AMP-FP16 / BF16.

Centralizes the "which autocast dtype, do we need a GradScaler" decision
in one place so Trainer doesn't have precision-string if/elif logic
scattered through the training loop.

RTX3050 strategy (per project hardware plan): start with "fp32" (no
autocast, no scaler) to prove correctness first, then move to "fp16"
(needs GradScaler — FP16 has limited dynamic range, so gradients can
underflow without loss scaling) once verified. H100 strategy: "bf16"
(same dynamic range as FP32, no GradScaler needed, and natively fast on
H100's tensor cores).
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

VALID_PRECISIONS = ("fp32", "fp16", "bf16")


class PrecisionManager:
    def __init__(self, precision: str = "fp32", device_type: str = "cuda") -> None:
        if precision not in VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {VALID_PRECISIONS}, got {precision!r}")
        self.precision = precision
        self.device_type = device_type

        import torch

        self.needs_scaler = precision == "fp16"
        self.scaler = torch.amp.GradScaler(device_type) if self.needs_scaler else None

        self._autocast_dtype = {
            "fp32": None,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]

    @contextmanager
    def autocast(self) -> Iterator[None]:
        """Context manager wrapping the forward pass with the right autocast (or none for fp32)."""
        import torch

        if self._autocast_dtype is None:
            yield  # fp32: no autocast at all
        else:
            with torch.autocast(device_type=self.device_type, dtype=self._autocast_dtype):
                yield

    def backward(self, loss) -> None:
        """Scaled backward for fp16, plain backward otherwise."""
        if self.needs_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, optimizer) -> None:
        """Scaled optimizer step (+ scaler update) for fp16, plain step otherwise."""
        if self.needs_scaler:
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            optimizer.step()

    def unscale_(self, optimizer) -> None:
        """Must be called before gradient clipping under fp16 (no-op otherwise)."""
        if self.needs_scaler:
            self.scaler.unscale_(optimizer)

    def state_dict(self) -> dict | None:
        return self.scaler.state_dict() if self.needs_scaler else None

    def load_state_dict(self, state: dict | None) -> None:
        if self.needs_scaler and state is not None:
            self.scaler.load_state_dict(state)
