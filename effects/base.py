"""
Effect framework — optional post-processing on the woven output.

The core operation of this tool is the time-displacement weave (weave.py); the
effects here (feedback trails, the blur suite) are filters applied AFTER it.
An Effect transforms a *list* of (T, H, W, 3) uint8 RGB frame stacks — in
practice a single-element list holding the weave result. Working in whole
sequences (not per-pixel, not per-frame-in-Python) keeps everything vectorised —
the hard rule for this project.

Effects declare a PARAMS schema (name/label/type/range/default per param), so
new effects are self-describing: subclass Effect, decorate with @register,
import the module in effects/__init__.py.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core import RenderContext


class Effect:
    name: str = "base"
    label: str = "Base"
    reduces: bool = False          # legacy flag, kept for subclass compat
    PARAMS: list[dict[str, Any]] = []   # UI/param schema (see blur.py for examples)

    def __init__(self, **params: Any):
        merged = {p["name"]: p.get("default") for p in self.PARAMS}
        merged.update({k: v for k, v in params.items() if k in merged})
        self.params = merged

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        """Override me. Return a new list of layer arrays."""
        return layers

    # convenience -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


# ── Registry ────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[Effect]] = {}


def register(cls: type[Effect]) -> type[Effect]:
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[type[Effect]]:
    return list(_REGISTRY.values())


def get(name: str) -> type[Effect]:
    return _REGISTRY[name]


def build(name: str, params: dict[str, Any] | None = None) -> Effect:
    return _REGISTRY[name](**(params or {}))


# ── Small shared numeric helpers ─────────────────────────────────────────────────
def to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)
