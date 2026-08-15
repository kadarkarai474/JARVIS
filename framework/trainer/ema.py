"""Exponential Moving Average (EMA) of model weights.

Maintains a shadow copy of every parameter and buffer, updated after each
optimizer step as `shadow = decay * shadow + (1 - decay) * param`. EMA
weights are used for evaluation/checkpointing (typically generalize
slightly better than the raw, noisier training weights), never for the
forward/backward pass used to compute gradients.

Includes the standard "decay warmup" trick (Karras et al. / common in
diffusion & GAN training): early in training, effective_decay is capped
lower so the shadow doesn't stay pinned near randomly-initialized weights
for too long. Disable via warmup=False for a plain fixed-decay EMA.
"""

from __future__ import annotations

import copy

import torch


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999, warmup: bool = True) -> None:
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        # Deep-copy state dict tensors so shadow storage is fully independent of the live model.
        self.shadow: dict[str, torch.Tensor] = {
            name: param.detach().clone() for name, param in model.state_dict().items()
        }

    def _effective_decay(self) -> float:
        if not self.warmup:
            return self.decay
        # Ramps from 0 towards self.decay as num_updates grows, capped at self.decay.
        return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        """Call once per optimizer step (after optimizer.step())."""
        decay = self._effective_decay()
        model_state = model.state_dict()
        for name, shadow_param in self.shadow.items():
            model_param = model_state[name]
            if shadow_param.dtype.is_floating_point:
                shadow_param.mul_(decay).add_(model_param.detach(), alpha=1 - decay)
            else:
                # Non-float buffers (e.g. int counters) — just copy, EMA doesn't apply.
                shadow_param.copy_(model_param)
        self.num_updates += 1

    def apply_shadow(self, model: torch.nn.Module) -> dict[str, torch.Tensor]:
        """Swap the model's live weights for the EMA shadow weights (e.g. before validation).

        Returns the original state dict so `restore()` can put it back afterward.
        """
        backup = copy.deepcopy(model.state_dict())
        model.load_state_dict(self.shadow)
        return backup

    def restore(self, model: torch.nn.Module, backup: dict[str, torch.Tensor]) -> None:
        """Restore the live training weights after apply_shadow() + evaluation."""
        model.load_state_dict(backup)

    def state_dict(self) -> dict:
        """For checkpointing — the shadow weights plus enough to resume warmup correctly."""
        return {"shadow": self.shadow, "num_updates": self.num_updates, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.shadow = state["shadow"]
        self.num_updates = state["num_updates"]
        self.decay = state["decay"]
