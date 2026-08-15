"""Denoising task variant: same-resolution output (scale_factor=1).

Added alongside restoration_sr and super_resolution so the framework
genuinely supports same-resolution restoration problems, not just the
2x-upscaling case — proves the task abstraction isn't secretly hardcoded
to scale=2 anywhere.
"""

from __future__ import annotations

from framework.registry import TASK_REGISTRY
from tasks.base import BaseTask


@TASK_REGISTRY.register("denoising")
class DenoisingTask(BaseTask):
    def __init__(self) -> None:
        super().__init__(
            name="denoising",
            scale_factor=1,
            input_channels=1,
            output_channels=1,
            default_loss_cfg={"name": "charbonnier"},
            default_metric_names=["psnr", "ssim"],
        )
