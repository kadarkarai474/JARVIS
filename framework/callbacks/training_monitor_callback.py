"""Per-epoch training monitor.

Reports:
- train/validation loss
- PSNR, SSIM, LPIPS, MSE
- single-image end-to-end latency
- model-only latency
- throughput
- parameters
- FLOPs
- GPU memory
- best metrics
- early stopping
- CSV history
"""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from framework.callbacks.base import Callback
from framework.registry import CALLBACK_REGISTRY
from framework.trainer.step import compute_step


@CALLBACK_REGISTRY.register("training_monitor")
class TrainingMonitorCallback(Callback):

    def __init__(
        self,
        save_dir: str,
        patience: int = 5,
        monitor: str = "val_loss",
        mode: str = "min",
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.patience = patience
        self.monitor = monitor
        self.mode = mode

        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = -1
        self.bad_epochs = 0

        self.csv_path = self.save_dir / "metrics.csv"
        self._csv_initialized = False

        self.parameters_m = None
        self.flops_g = None

    def _improved(self, value: float) -> bool:
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def _count_parameters(self, model: Any) -> float:
        return sum(p.numel() for p in model.parameters()) / 1e6

    def _calculate_flops(self, model: Any, device: str) -> float | None:
        try:
            from thop import profile
        except ImportError:
            return None

        try:
            was_training = model.training
            model.eval()

            dummy = torch.zeros(
                1, 1, 128, 128,
                device=device,
                dtype=torch.float32,
            )

            with torch.no_grad():
                flops, _ = profile(
                    model,
                    inputs=(dummy,),
                    verbose=False,
                )

            if was_training:
                model.train()

            return float(flops) / 1e9

        except Exception as exc:
            print(f"[WARN] FLOPs calculation failed: {exc}")
            return None

    def _benchmark_inference(
        self,
        model: Any,
        val_loader: Any,
        device: str,
        task: Any,
    ) -> dict[str, float]:

        if val_loader is None:
            return {}

        # ------------------------------------------------------------
        # Data loading
        # ------------------------------------------------------------
        if "cuda" in str(device):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        batch = next(iter(val_loader))
        data_load_ms = (time.perf_counter() - t0) * 1000.0

        # ------------------------------------------------------------
        # CPU -> GPU
        # ------------------------------------------------------------
        if "cuda" in str(device):
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        batch_dev = {
            k: (v.to(device, non_blocking=True) if hasattr(v, "to") else v)
            for k, v in batch.items()
        }

        if "cuda" in str(device):
            torch.cuda.synchronize()

        transfer_ms = (time.perf_counter() - t1) * 1000.0

        # ------------------------------------------------------------
        # Model inference
        # ------------------------------------------------------------
        model.eval()

        if "cuda" in str(device):
            torch.cuda.synchronize()

        t2 = time.perf_counter()

        with torch.no_grad():
            result = compute_step(
                model,
                batch_dev,
                task,
                loss_fn=None,
            )
            prediction = result["prediction"]

        if "cuda" in str(device):
            torch.cuda.synchronize()

        inference_ms = (time.perf_counter() - t2) * 1000.0

        # ------------------------------------------------------------
        # GPU -> CPU / output ready
        # ------------------------------------------------------------
        if "cuda" in str(device):
            torch.cuda.synchronize()

        t3 = time.perf_counter()

        _ = prediction.detach().cpu()

        output_ms = (time.perf_counter() - t3) * 1000.0

        total_ms = (
            data_load_ms
            + transfer_ms
            + inference_ms
            + output_ms
        )

        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        return {
            "data_load_ms": data_load_ms,
            "transfer_ms": transfer_ms,
            "inference_ms": inference_ms,
            "output_ms": output_ms,
            "end_to_end_ms": total_ms,
            "fps": fps,
        }

    def _gpu_memory(self, device: str) -> tuple[float, float]:
        if not torch.cuda.is_available() or "cuda" not in str(device):
            return 0.0, 0.0

        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        return allocated, reserved

    def _write_csv(self, row: dict[str, Any]) -> None:

        fields = list(row.keys())

        write_header = not self.csv_path.exists()

        with self.csv_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            if write_header:
                writer.writeheader()

            writer.writerow(row)

    def on_train_start(self, state: dict[str, Any]) -> None:

        model = state["model"]
        device = state["device"]

        self.parameters_m = self._count_parameters(model)
        self.flops_g = self._calculate_flops(model, device)

        print()
        print("=" * 78)
        print("RESTORMER TRAINING MONITOR")
        print("=" * 78)

        print(
            f"Parameters : {self.parameters_m:.3f} M"
        )

        if self.flops_g is not None:
            print(
                f"FLOPs      : {self.flops_g:.3f} GFLOPs "
                f"(128x128 input)"
            )
        else:
            print("FLOPs      : unavailable")

        print("Expected   : 1x128x128 -> 1x256x256")
        print("=" * 78)

    def on_epoch_end(self, state: dict[str, Any]) -> None:

        epoch = int(state["epoch"])
        total_epochs = int(state.get("num_epochs", 0))

        train_metrics = state.get("train_metrics", {})
        val_metrics = state.get("val_metrics", {})

        train_loss = train_metrics.get("loss", float("nan"))
        val_loss = val_metrics.get("loss", float("nan"))

        psnr = val_metrics.get("psnr", float("nan"))
        ssim = val_metrics.get("ssim", float("nan"))
        lpips = val_metrics.get("lpips", float("nan"))
        mse = val_metrics.get("mse", float("nan"))

        benchmark = self._benchmark_inference(
            state["model"],
            state.get("val_loader"),
            state["device"],
            state["task"],
        )

        allocated, reserved = self._gpu_memory(state["device"])

        current_value = val_loss

        improved = False

        if current_value == current_value:
            if self._improved(current_value):
                self.best_value = current_value
                self.best_epoch = epoch
                self.bad_epochs = 0
                improved = True
            else:
                self.bad_epochs += 1

        quality = (
            max(0.0, min(100.0, psnr / 35.0 * 100.0))
            if psnr == psnr
            else float("nan")
        )

        print()
        print("=" * 78)
        print(
            f"EPOCH {epoch + 1}"
            + (f"/{total_epochs}" if total_epochs else "")
        )
        print("=" * 78)

        print("TRAIN")
        print(f"  Loss                 : {train_loss:.6f}")

        print("-" * 78)

        print("VALIDATION")
        print(f"  Loss                 : {val_loss:.6f}")
        print(f"  PSNR                 : {psnr:.4f} dB")
        print(f"  SSIM                 : {ssim:.6f}")
        print(f"  LPIPS                : {lpips:.6f}")
        print(f"  MSE                  : {mse:.8f}")
        print(f"  Quality indicator*   : {quality:.2f}%")

        print("-" * 78)

        print("SINGLE IMAGE END-TO-END")
        if benchmark:
            print(
                f"  Data loading        : "
                f"{benchmark['data_load_ms']:.2f} ms"
            )
            print(
                f"  CPU -> GPU          : "
                f"{benchmark['transfer_ms']:.2f} ms"
            )
            print(
                f"  Model inference     : "
                f"{benchmark['inference_ms']:.2f} ms"
            )
            print(
                f"  GPU -> CPU          : "
                f"{benchmark['output_ms']:.2f} ms"
            )
            print(
                f"  TOTAL               : "
                f"{benchmark['end_to_end_ms']:.2f} ms"
            )
            print(
                f"  Throughput          : "
                f"{benchmark['fps']:.2f} images/sec"
            )

        print("-" * 78)

        print("MODEL")

        print(
            f"  Parameters          : "
            f"{self.parameters_m:.3f} M"
        )

        if self.flops_g is not None:
            print(
                f"  FLOPs               : "
                f"{self.flops_g:.3f} GFLOPs"
            )

        print("  Input               : 1 x 128 x 128")
        print("  Expected output     : 1 x 256 x 256")

        print("-" * 78)

        print("GPU")

        if torch.cuda.is_available():
            print(
                f"  Allocated VRAM      : "
                f"{allocated:.2f} GB"
            )
            print(
                f"  Reserved VRAM       : "
                f"{reserved:.2f} GB"
            )
        else:
            print("  CUDA                : unavailable")

        print("-" * 78)

        print("BEST / EARLY STOPPING")

        if self.best_epoch >= 0:
            print(
                f"  Best val loss       : "
                f"{self.best_value:.6f} "
                f"(epoch {self.best_epoch + 1})"
            )

        print(
            f"  Patience            : "
            f"{self.bad_epochs}/{self.patience}"
        )

        if improved:
            print("  Status              : ✓ VALIDATION IMPROVED")
        else:
            print("  Status              : no improvement")

        if self.bad_epochs >= self.patience:
            print()
            print(
                f"EARLY STOPPING: validation loss did not improve "
                f"for {self.patience} epochs."
            )
            state["stop_training"] = True

        print("=" * 78)

        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "psnr": psnr,
            "ssim": ssim,
            "lpips": lpips,
            "mse": mse,
            "quality_indicator": quality,
            "data_load_ms": benchmark.get("data_load_ms", ""),
            "transfer_ms": benchmark.get("transfer_ms", ""),
            "inference_ms": benchmark.get("inference_ms", ""),
            "output_ms": benchmark.get("output_ms", ""),
            "end_to_end_ms": benchmark.get("end_to_end_ms", ""),
            "fps": benchmark.get("fps", ""),
            "parameters_m": self.parameters_m,
            "flops_g": self.flops_g or "",
            "gpu_allocated_gb": allocated,
            "gpu_reserved_gb": reserved,
            "best_val_loss": self.best_value,
            "best_epoch": self.best_epoch + 1,
            "patience_count": self.bad_epochs,
        }

        self._write_csv(row)

        # Restore training mode after benchmark.
        state["model"].train()

    def on_train_end(self, state: dict[str, Any]) -> None:
        print()
        print("=" * 78)
        print("TRAINING MONITOR COMPLETE")
        print(f"Metrics CSV: {self.csv_path}")
        if self.best_epoch >= 0:
            print(f"Best epoch: {self.best_epoch + 1}")
            print(f"Best validation loss: {self.best_value:.6f}")
        print("=" * 78)
