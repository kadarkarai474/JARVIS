"""Blind restoration + 2x super-resolution — the primary hackathon task.

128x128 grayscale input degraded by an unknown combination of Gaussian
noise, speckle noise, blur, and downsampling artifacts -> 256x256 clean
grayscale output.
"""

from __future__ import annotations

from framework.registry import TASK_REGISTRY
from tasks.base import BaseTask


@TASK_REGISTRY.register("restoration_sr")
class BlindRestorationSRTask(BaseTask):
    """The project's default/primary task: scale_factor=2, degradation-agnostic."""

    def __init__(self) -> None:
        super().__init__(
            name="restoration_sr",
            scale_factor=2,
            input_channels=1,
            output_channels=1,
            default_loss_cfg={"name": "l1"},
            default_metric_names=["psnr", "ssim", "lpips"],
        )
