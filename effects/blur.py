"""
Blur suite (Phase 4).

Pluggable Effect subclasses. Each operates on the (already-combined) frame stack
as a *filter* — the temporal smear / soft finish on the woven result. All are
vectorised at the array level (frame loops are fine; per-pixel Python loops are
not).

Implemented:
  • TemporalAverageEffect  — long-exposure trails: mean of k consecutive frames.
  • MotionFlowBlurEffect   — motion-compensated directional blur (Farneback flow).
  • GaussianBlurEffect     — spatial Gaussian softening.
  • FFTLowPassEffect       — frequency-domain low-pass.
  • LuminanceGuidedBlurEffect — luminance controls blur amount (quantised levels).

Extension points (left deliberately open — subclass Effect + @register):
  • edge-guided bilateral, anisotropic diffusion, depth-based, dynamic-kernel,
    full spatio-temporal 3-D kernels.
  • cross-source variants (dissolve two sources' spectra; layer-A luminance
    driving layer-B blur) — run these *before* the combine stage, where the
    layer list still has 2+ members.
"""
from __future__ import annotations

import cv2
import numpy as np

from core import RenderContext
from effects.base import Effect, register, to_u8


@register
class TemporalAverageEffect(Effect):
    name = "temporal_average"
    label = "Blur · Temporal average (long exposure)"
    reduces = False
    PARAMS = [
        {"name": "window", "label": "Window (frames)", "type": "int",
         "min": 1, "max": 24, "step": 1, "default": 4},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        k = max(1, int(self.params["window"]))
        if k <= 1:
            return layers
        out = []
        for layer in layers:
            f = layer.astype(np.float32)
            pad_l, pad_r = k // 2, k - 1 - (k // 2)
            fp = np.pad(f, ((pad_l, pad_r), (0, 0), (0, 0), (0, 0)), mode="edge")
            cs = np.cumsum(fp, axis=0)
            cs = np.concatenate([np.zeros((1,) + fp.shape[1:], np.float32), cs], 0)
            win = cs[k:] - cs[:-k]          # sliding window sums, length T
            out.append(to_u8(win / k))
        return out


@register
class MotionFlowBlurEffect(Effect):
    name = "motion_flow_blur"
    label = "Blur · Motion-compensated (optical flow)"
    reduces = False
    PARAMS = [
        {"name": "taps", "label": "Taps per side", "type": "int",
         "min": 1, "max": 6, "step": 1, "default": 3},
        {"name": "strength", "label": "Flow strength", "type": "float",
         "min": 0.2, "max": 4.0, "step": 0.1, "default": 1.0},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        taps = max(1, int(self.params["taps"]))
        strength = float(self.params["strength"])
        out = []
        for layer in layers:
            f = layer
            T, H, W = f.shape[:3]
            gray = [cv2.cvtColor(f[t], cv2.COLOR_RGB2GRAY) for t in range(T)]
            base_x, base_y = np.meshgrid(
                np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
            res = np.empty_like(f, dtype=np.float32)
            for t in range(T):
                a, b = (gray[0], gray[min(1, T - 1)]) if t == 0 else (gray[t - 1], gray[t])
                flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                fx, fy = flow[..., 0] * strength, flow[..., 1] * strength
                acc = f[t].astype(np.float32).copy()
                count = 1.0
                for s in range(1, taps + 1):
                    frac = s / float(taps)
                    for sign in (-1.0, 1.0):
                        mx = (base_x + sign * frac * fx).astype(np.float32)
                        my = (base_y + sign * frac * fy).astype(np.float32)
                        acc += cv2.remap(f[t], mx, my, cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT).astype(np.float32)
                        count += 1.0
                res[t] = acc / count
            out.append(to_u8(res))
        return out


@register
class GaussianBlurEffect(Effect):
    name = "gaussian_blur"
    label = "Blur · Gaussian (spatial)"
    reduces = False
    PARAMS = [
        {"name": "sigma", "label": "Sigma (px)", "type": "float",
         "min": 0.0, "max": 20.0, "step": 0.5, "default": 3.0},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        sigma = float(self.params["sigma"])
        if sigma <= 0.01:
            return layers
        out = []
        for layer in layers:
            res = np.empty_like(layer)
            for t in range(layer.shape[0]):
                res[t] = cv2.GaussianBlur(layer[t], (0, 0), sigmaX=sigma, sigmaY=sigma)
            out.append(res)
        return out


@register
class FFTLowPassEffect(Effect):
    name = "fft_lowpass"
    label = "Blur · Frequency-domain low-pass"
    reduces = False
    PARAMS = [
        {"name": "cutoff", "label": "Cutoff (keep fraction)", "type": "float",
         "min": 0.02, "max": 1.0, "step": 0.02, "default": 0.25},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        cutoff = float(self.params["cutoff"])
        if cutoff >= 0.999:
            return layers
        out = []
        for layer in layers:
            T, H, W = layer.shape[:3]
            cy, cx = H / 2.0, W / 2.0
            yy, xx = np.ogrid[:H, :W]
            r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
            mask = (r <= cutoff).astype(np.float32)[..., None]   # (H,W,1) low-pass
            res = np.empty_like(layer)
            for t in range(T):
                fr = layer[t].astype(np.float32)
                spec = np.fft.fftshift(np.fft.fft2(fr, axes=(0, 1)), axes=(0, 1))
                spec *= mask
                rec = np.fft.ifft2(np.fft.ifftshift(spec, axes=(0, 1)), axes=(0, 1)).real
                res[t] = to_u8(rec)
            out.append(res)
        return out


@register
class LuminanceGuidedBlurEffect(Effect):
    name = "luminance_blur"
    label = "Blur · Luminance-guided"
    reduces = False
    PARAMS = [
        {"name": "max_sigma", "label": "Max sigma (px)", "type": "float",
         "min": 1.0, "max": 24.0, "step": 1.0, "default": 8.0},
        {"name": "levels", "label": "Blur levels", "type": "int",
         "min": 2, "max": 8, "step": 1, "default": 5},
        {"name": "invert", "label": "Blur darks instead", "type": "bool",
         "default": False},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        max_sigma = float(self.params["max_sigma"])
        levels = max(2, int(self.params["levels"]))
        invert = bool(self.params["invert"])
        out = []
        for layer in layers:
            res = np.empty_like(layer)
            for t in range(layer.shape[0]):
                frame = layer[t]
                lum = (0.299 * frame[..., 0] + 0.587 * frame[..., 1]
                       + 0.114 * frame[..., 2])
                if invert:
                    lum = 255.0 - lum
                bucket = np.clip((lum / 256.0 * levels).astype(np.int32), 0, levels - 1)
                acc = frame.copy()
                for lv in range(levels):
                    sigma = (lv / (levels - 1)) * max_sigma
                    blurred = (frame if sigma <= 0.01
                               else cv2.GaussianBlur(frame, (0, 0), sigmaX=sigma, sigmaY=sigma))
                    np.copyto(acc, blurred, where=(bucket == lv)[..., None])
                res[t] = acc
            out.append(res)
        return out
