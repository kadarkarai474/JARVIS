"""Pure super-resolution task variant.

Same scale_factor as the primary task (2x), but conceptually assumes a
cleaner input (no heavy noise/blur to also remove) — kept as a distinct,
separately-configurable task so future experiments (e.g. ablating "does
denoising-then-SR beat blind joint SR" style setups) don't require
touching the primary task's defaults.
"""

from __future__ import annotations

from framework.registry import TASK_REGISTRY
from tasks.base import BaseTask


@TASK_REGISTRY.register("super_resolution")
class SuperResolutionTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            name="super_resolution",
            scale_factor=2,
            input_channels=1,
            output_channels=1,
            default_loss_cfg={"name": "l1"},
            default_metric_names=["psnr", "ssim"],
        )
