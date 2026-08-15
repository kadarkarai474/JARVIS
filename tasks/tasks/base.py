"""Task abstraction.

A "task" fully describes the input/output contract and default training
recipe for one problem variant (blind restoration+SR, pure SR, denoising,
...). Trainer/evaluator/inference/benchmark code depends only on this
interface — never on a specific scale factor or resolution — so adding a
new task variant never requires touching those modules.

Design note: this module is intentionally torch-free. `validate_shapes`
works on plain (H, W) tuples, and `preprocess`/`postprocess` operate on
whatever object exposes `.shape` (numpy arrays or torch tensors both
qualify) — so task logic is unit-testable without a torch install, same as
utils/data_transforms.py in Phase 3.
"""

from __future__ import annotations

import abc
import dataclasses
from typing import Any


class ShapeContractError(ValueError):
    """Raised when an (input, target) shape pair doesn't match a task's contract."""


@dataclasses.dataclass
class BaseTask(abc.ABC):
    """Abstract base for all tasks.

    Attributes:
        name: Registry key for this task (matches TASK_REGISTRY registration).
        scale_factor: output_size / input_size along each spatial dim.
            1 for same-resolution tasks (denoising), 2 for the primary
            blind-restoration+SR task and pure super-resolution.
        input_channels: Expected input channel count (1 = grayscale, per spec).
        output_channels: Expected output channel count.
        default_loss_cfg: Default loss registry config for this task, e.g.
            {"name": "l1"}. Concrete loss names are only guaranteed to
            resolve once Phase 5 registers them — storing the name here now
            doesn't require it to exist yet.
        default_metric_names: Default metric registry names for this task.
    """

    name: str
    scale_factor: int
    input_channels: int = 1
    output_channels: int = 1
    default_loss_cfg: dict = dataclasses.field(default_factory=lambda: {"name": "l1"})
    default_metric_names: list = dataclasses.field(default_factory=lambda: ["psnr", "ssim"])

    def validate_shapes(self, input_shape: tuple[int, int], target_shape: tuple[int, int]) -> None:
        """Verify an (input, target) spatial-shape pair satisfies this task's contract.

        Args:
            input_shape: (H, W) of the model input.
            target_shape: (H, W) of the ground truth / target.

        Raises:
            ShapeContractError: if target_shape != input_shape * scale_factor
                (element-wise).
        """
        expected = (input_shape[0] * self.scale_factor, input_shape[1] * self.scale_factor)
        if tuple(target_shape) != expected:
            raise ShapeContractError(
                f"Task '{self.name}' (scale_factor={self.scale_factor}) expected target shape "
                f"{expected} for input shape {input_shape}, got {tuple(target_shape)}."
            )

    def preprocess(self, batch: dict) -> dict:
        """Task-specific batch preprocessing hook, applied before the model forward pass.

        Default is a no-op passthrough — override for tasks that need special
        handling (e.g. a task that needs an explicit bicubic pre-upsample
        before a residual-learning model). Kept as a hook (not hardcoded in
        the trainer) specifically so new tasks never require trainer changes.
        """
        return batch

    def postprocess(self, model_output: Any, batch: dict) -> Any:
        """Task-specific output postprocessing hook, applied after the model forward pass.

        Default is a no-op passthrough.
        """
        return model_output

    def check_model_compatibility(self, model_scale_factor: int) -> None:
        """Verify a model's declared scale_factor matches this task's contract.

        Called at model-build time (task + model are both selected via
        config) so a mismatched pairing fails immediately and clearly,
        rather than producing silently-wrong-shaped output deep in training.
        """
        if model_scale_factor != self.scale_factor:
            raise ShapeContractError(
                f"Task '{self.name}' requires scale_factor={self.scale_factor}, but the "
                f"selected model declares scale_factor={model_scale_factor}. "
                "Pick a task/model pair with matching scale factors."
            )
