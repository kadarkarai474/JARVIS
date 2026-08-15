"""Generic component registry.

This is the single mechanism that makes models, losses, metrics, tasks,
optimizers, schedulers, and callbacks interchangeable through config alone.
Every one of those component families gets its own Registry instance
(see framework/registry/__init__.py) but shares this exact same class —
so "how do I add a new X" always has the same answer: implement it,
decorate it with @X_REGISTRY.register("name"), reference "name" in a config.

No other framework module (trainer, evaluator, inference, benchmark) should
ever import a concrete model/loss/metric/task class directly — they must
go through a registry's .build(cfg) method.
"""

from __future__ import annotations

from typing import Any, Callable, TypeVar

T = TypeVar("T")


class RegistryError(KeyError):
    """Raised for duplicate registration or unknown-name lookup.

    Subclasses KeyError (rather than a bare Exception) so existing
    `except KeyError` call sites still work, while still being specific
    enough to catch on its own.
    """


class Registry:
    """A name -> constructor mapping with config-driven instantiation.

    Example:
        MODEL_REGISTRY = Registry("model")

        @MODEL_REGISTRY.register("nafnet")
        class NAFNet(BaseRestorationModel):
            ...

        model = MODEL_REGISTRY.build({"name": "nafnet", "params": {"width": 32}})
    """

    def __init__(self, kind: str) -> None:
        """Args:
        kind: Human-readable label for this registry family (used only in
            error messages), e.g. "model", "loss", "metric", "task".
        """
        self._kind = kind
        self._obj_map: dict[str, Callable[..., Any]] = {}

    def register(self, name: str | None = None) -> Callable[[T], T]:
        """Class/function decorator that registers the target under `name`.

        If `name` is omitted, the decorated object's `__name__` (lowercased)
        is used instead.
        """

        def decorator(obj: T) -> T:
            key = name if name is not None else obj.__name__.lower()  # type: ignore[union-attr]
            if key in self._obj_map:
                raise RegistryError(
                    f"'{key}' is already registered in the {self._kind} registry "
                    f"(existing: {self._obj_map[key]!r}, new: {obj!r}). "
                    "Registry keys must be unique — pick a different name."
                )
            self._obj_map[key] = obj
            return obj

        return decorator

    def get(self, name: str) -> Callable[..., Any]:
        """Look up a registered constructor by name.

        Raises:
            RegistryError: if `name` isn't registered, listing what IS
                available so the error is actionable, not just "KeyError".
        """
        if name not in self._obj_map:
            available = ", ".join(sorted(self._obj_map)) or "(nothing registered yet)"
            raise RegistryError(
                f"Unknown {self._kind} '{name}'. Available: {available}"
            )
        return self._obj_map[name]

    def build(self, cfg: Any, **extra_kwargs: Any) -> Any:
        """Instantiate a registered component from a config.

        `cfg` may be a plain dict or an OmegaConf DictConfig (both support
        the same `cfg["name"]` / `cfg.get(...)` access pattern used here) —
        this function has no OmegaConf import so it works identically
        before and after Hydra is wired in.

        Expected shape: {"name": <registry key>, "params": {...}}  (params optional)

        Args:
            cfg: Config with at least a "name" key.
            **extra_kwargs: Additional keyword arguments merged in on top of
                cfg["params"] — used e.g. by tasks to inject task-derived
                values (like scale_factor) into a model's constructor
                without the config needing to duplicate them.

        Returns:
            The constructed object.
        """
        if "name" not in cfg:
            raise ValueError(f'{self._kind} config must contain a "name" key, got: {cfg}')
        name = cfg["name"]
        params = dict(cfg.get("params", {}) or {})
        params.update(extra_kwargs)
        ctor = self.get(name)
        return ctor(**params)

    def list_registered(self) -> list[str]:
        """Return all registered names, sorted — useful for CLI help text and tests."""
        return sorted(self._obj_map)

    def __contains__(self, name: str) -> bool:
        return name in self._obj_map

    def __len__(self) -> int:
        return len(self._obj_map)

    def __repr__(self) -> str:
        return f"Registry(kind={self._kind!r}, registered={self.list_registered()!r})"
