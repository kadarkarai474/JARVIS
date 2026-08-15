"""Shared forward(+loss) step — the concrete proof that trainer/evaluator
logic is task- and model-agnostic.

Both the training loop (Phase 7) and the evaluation loop (Phase 8) call
this same function. It only ever talks to `model`, `task`, and `loss_fn`
through their abstract interfaces (BaseRestorationModel.forward,
BaseTask.preprocess/postprocess, BaseLoss.forward) — never a concrete
class — so swapping NAFNet for Restormer, or restoration_sr for denoising,
never requires touching this function.
"""

from __future__ import annotations

from typing import Any

from tasks.base import BaseTask


def compute_step(
    model: Any,
    batch: dict,
    task: BaseTask,
    loss_fn: Any | None = None,
) -> dict:
    """Run one forward pass (and optionally loss) for any (model, task) pair.

    Args:
        model: Any BaseRestorationModel — only `model(x)` is called.
        batch: Dict with at least a "noisy" key (and "gt" if loss_fn is given).
        task: The active BaseTask — supplies preprocess/postprocess hooks.
        loss_fn: Optional BaseLoss. If provided, batch must contain "gt" and
            the returned dict includes a "loss" entry.

    Returns:
        {"prediction": <model output, post-processed>, "loss": <scalar tensor>?}
    """
    processed_batch = task.preprocess(batch)
    prediction = model(processed_batch["noisy"])
    prediction = task.postprocess(prediction, processed_batch)

    result = {"prediction": prediction}
    if loss_fn is not None:
        if "gt" not in processed_batch:
            raise ValueError("loss_fn was provided but batch has no 'gt' key")
        result["loss"] = loss_fn(prediction, processed_batch["gt"])
    return result
