"""
Small cross-cutting helpers: device detection and a render context.

Torch is *optional*. If it (and a CUDA GPU) are present we report "cuda" so
effects can opt into acceleration; otherwise everything runs on numpy/OpenCV so
a collaborator with no GPU (and no torch install) can run the whole tool.
"""
from __future__ import annotations

from dataclasses import dataclass


def get_device() -> str:
    """Return 'cuda' if torch + a CUDA GPU are available, else 'cpu'.

    Never raises — a missing torch is the normal, supported path.
    """
    try:
        import torch  # noqa: WPS433 (optional dep, imported lazily on purpose)

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def device_label() -> str:
    dev = get_device()
    if dev == "cuda":
        try:
            import torch

            return f"CUDA ({torch.cuda.get_device_name(0)})"
        except Exception:
            return "CUDA"
    return "CPU (numpy/OpenCV)"


@dataclass
class RenderContext:
    """Passed down the effect chain so effects can adapt to preview vs. full
    render and to the available compute device.

    `layer_opacities` is aligned with the *source* layer list the compositor
    built; combiner effects (e.g. superimpose) read it for per-layer opacity.
    """

    fps: int
    width: int
    height: int
    is_preview: bool = False
    device: str = "cpu"
    layer_opacities: list[float] | None = None
