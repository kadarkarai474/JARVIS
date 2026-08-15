"""Concrete Inferencer: full implementation of BaseInferencer (Phase 4).

This is the actual submission-generation pathway for the test/ split
(NoisyLR only, no GT — per the project's dataset contract). Built on
compute_step with loss_fn=None, same as the Evaluator, but with no metric
computation at all (there's no GT to score against) and with a save-to-disk
path since the whole point of inference is producing files a judge/grader
can consume.

Output convention: predictions are saved as .npy, float32, matching the
project's dataset format exactly — so a predicted 256x256 output for
stem "0007" is saved as "0007.npy", directly comparable to a GT/ file if
one ever becomes available for scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from framework.inference.base import BaseInferencer
from framework.trainer.step import compute_step
from tasks.base import BaseTask


def _move_batch_to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


class Inferencer(BaseInferencer):
    def __init__(
        self,
        task: BaseTask,
        model: Any,
        device: str = "cuda",
        output_dir: str | Path | None = None,
        config: dict | None = None,
    ) -> None:
        super().__init__(task, model, config)
        self.device = device
        self.output_dir = Path(output_dir) if output_dir is not None else None

    def run(self, data_loader: Any) -> list[dict[str, Any]]:
        """Run inference over every batch in `data_loader`.

        Returns a list of {"stem": str, "prediction": np.ndarray} for every
        sample. If `self.output_dir` was set, each prediction is also
        written to `{output_dir}/{stem}.npy` (float32, matching the
        dataset's on-disk format).
        """
        import torch

        self.model.eval()
        results: list[dict[str, Any]] = []

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch in data_loader:
                batch_dev = _move_batch_to_device(batch, self.device)
                result = compute_step(self.model, batch_dev, self.task, loss_fn=None)
                pred = result["prediction"]
                stems = batch.get("stem", [str(i) for i in range(pred.shape[0])])

                for b, stem in enumerate(stems):
                    pred_np = pred[b, 0].detach().cpu().numpy().astype(np.float32)
                    if self.output_dir is not None:
                        np.save(self.output_dir / f"{stem}.npy", pred_np)
                    results.append({"stem": stem, "prediction": pred_np})

        return results

    def run_single(self, noisy: Any) -> np.ndarray:
        """Run inference on one in-memory NoisyLR array/tensor.

        Args:
            noisy: (H, W) or (1, H, W) array/tensor, already normalized the
                same way training data was (e.g. via utils.data_transforms
                .normalize with the same mode used at train time — this
                function does NOT re-normalize, to stay a pure model-call
                wrapper with no hidden assumptions about input scale).

        Returns:
            (H*scale, W*scale) float32 numpy array.
        """
        import torch

        self.model.eval()

        if not torch.is_tensor(noisy):
            noisy = torch.as_tensor(np.asarray(noisy, dtype=np.float32))
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(0)  # (H,W) -> (1,H,W)
        noisy = noisy.unsqueeze(0).to(self.device)  # -> (1,1,H,W)

        with torch.no_grad():
            batch = self.task.preprocess({"noisy": noisy})
            prediction = self.model(batch["noisy"])
            prediction = self.task.postprocess(prediction, batch)

        return prediction[0, 0].detach().cpu().numpy().astype(np.float32)
