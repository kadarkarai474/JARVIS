"""Concrete Evaluator: full implementation of BaseEvaluator (Phase 4).

Runs the exact same compute_step used by Phase 7's Trainer (with
loss_fn=None, since evaluation doesn't need a loss value) — this is the
concrete payoff of that shared-step design: the evaluator required zero
new forward-pass logic, only metric accumulation and reporting around it.

Two-pass design for visualization: `_run()` accumulates all metrics and
per-image scores in one pass over the full split WITHOUT holding every
prediction image in memory (important for a large real test set — only
per-sample scalar scores are kept, not images). `save_report()` then
does a second, cheap pass that re-runs the model only for the identified
worst-K samples to fetch actual images for visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from framework.evaluator.base import BaseEvaluator
from framework.evaluator.report import (
    find_best_samples,
    find_worst_samples,
    write_per_image_csv,
    write_summary_markdown,
)
from framework.evaluator.visualize import plot_comparison_grid, plot_difference_map
from framework.benchmark.report import write_benchmark_json
from framework.trainer.step import compute_step
from tasks.base import BaseTask


def _move_batch_to_device(batch: dict, device: str) -> dict:
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}


class Evaluator(BaseEvaluator):
    def __init__(
        self,
        task: BaseTask,
        model: Any,
        metrics: dict[str, Any],
        data_loader: Any,
        device: str = "cuda",
        config: dict | None = None,
    ) -> None:
        super().__init__(task, model, metrics, data_loader, config)
        self.device = device
        self._stems: list[str] = []
        self._has_run = False

    def _run(self) -> None:
        if self._has_run:
            return
        import torch

        self.model.eval()
        for m in self.metrics.values():
            m.reset()
        self._stems = []

        with torch.no_grad():
            for batch in self.data_loader:
                if "gt" not in batch:
                    raise ValueError(
                        "Evaluator requires ground truth ('gt' key in batch) — use the "
                        "Inferencer (Phase 9) for GT-free splits like 'test'."
                    )
                batch_dev = _move_batch_to_device(batch, self.device)
                result = compute_step(self.model, batch_dev, self.task, loss_fn=None)
                pred, gt = result["prediction"], batch_dev["gt"]
                stems = batch.get("stem", [str(i) for i in range(pred.shape[0])])

                for b in range(pred.shape[0]):
                    for m in self.metrics.values():
                        m.update(pred[b, 0], gt[b, 0])
                    self._stems.append(stems[b])

        self._has_run = True

    def evaluate(self) -> dict[str, float]:
        self._run()
        return {name: m.compute() for name, m in self.metrics.items()}

    def evaluate_per_image(self) -> list[dict[str, Any]]:
        self._run()
        n = len(self._stems)
        rows = []
        for i in range(n):
            row: dict[str, Any] = {"stem": self._stems[i]}
            for name, m in self.metrics.items():
                values = getattr(m, "values", None)
                row[name] = values[i] if values is not None and len(values) == n else None
            rows.append(row)
        return rows

    def _fetch_images_for_stems(self, target_stems: set[str]) -> dict[str, dict]:
        """Second pass: re-run the model only for the requested stems, to get
        actual (noisy, prediction, gt) arrays for visualization without
        having kept every image in memory during the main metric pass."""
        import torch

        found: dict[str, dict] = {}
        with torch.no_grad():
            for batch in self.data_loader:
                stems = batch.get("stem", [])
                if not any(s in target_stems for s in stems):
                    continue
                batch_dev = _move_batch_to_device(batch, self.device)
                result = compute_step(self.model, batch_dev, self.task, loss_fn=None)
                pred = result["prediction"]
                for b, stem in enumerate(stems):
                    if stem in target_stems:
                        found[stem] = {
                            "stem": stem,
                            "noisy": batch["noisy"][b, 0].cpu().numpy(),
                            "prediction": pred[b, 0].detach().cpu().numpy(),
                            "gt": batch["gt"][b, 0].cpu().numpy(),
                        }
                if len(found) == len(target_stems):
                    break
        return found

    def save_report(
        self,
        output_dir: str | Path,
        primary_metric: str = "psnr",
        primary_metric_mode: str = "min",
        k_worst: int = 10,
        k_best: int = 10,
        split_name: str = "val",
    ) -> dict[str, Any]:
        """Write the full evaluation report: per-image CSV, Markdown summary,
        worst-K and best-K comparison grids + difference maps.
        Returns the aggregate metrics dict.
        """
        output_dir = Path(output_dir)
        aggregate = self.evaluate()
        per_image_rows = self.evaluate_per_image()

        write_per_image_csv(
            per_image_rows,
            output_dir / "per_image_metrics.csv",
        )
        write_benchmark_json(
            aggregate,
            output_dir / "aggregate_metrics.json",
        )

        worst_rows = find_worst_samples(
            per_image_rows,
            primary_metric,
            k=k_worst,
            mode=primary_metric_mode,
        )

        best_rows = find_best_samples(
            per_image_rows,
            primary_metric,
            k=k_best,
            mode=primary_metric_mode,
        )

        write_summary_markdown(
            aggregate,
            per_image_rows,
            worst_rows,
            primary_metric,
            output_dir / "REPORT.md",
            split_name,
        )

        if worst_rows:
            worst_stems = {r["stem"] for r in worst_rows}
            images_by_stem = self._fetch_images_for_stems(worst_stems)

            ordered_samples = [
                images_by_stem[r["stem"]]
                for r in worst_rows
                if r["stem"] in images_by_stem
            ]

            plot_comparison_grid(
                ordered_samples,
                output_dir / "images" / "worst_samples_comparison.png",
                title=f"Worst {len(ordered_samples)} samples by {primary_metric}",
            )

            for sample in ordered_samples:
                plot_difference_map(
                    sample["prediction"],
                    sample["gt"],
                    output_dir / "images" / f"diff_{sample['stem']}.png",
                )

        if best_rows:
            best_stems = {r["stem"] for r in best_rows}
            images_by_stem = self._fetch_images_for_stems(best_stems)

            ordered_samples = [
                images_by_stem[r["stem"]]
                for r in best_rows
                if r["stem"] in images_by_stem
            ]

            plot_comparison_grid(
                ordered_samples,
                output_dir / "images" / "best_samples_comparison.png",
                title=f"Best {len(ordered_samples)} samples by {primary_metric}",
            )

        return aggregate