"""
Blur suite — streaming post stages, usable on ANY generator's output including
plain frames (no weave, no superimposition involved).

Each stage takes one `amount` in 0–1 driving its primary parameter, streams with
bounded buffers, and is vectorised (no per-pixel Python loops).

    smear              fixed-angle directional motion blur       (angle + amount → 1–48px)
    temporal_average   ring buffer mean — long exposure          (amount → window 1–24)
    motion_flow        Farneback flow, blur along the vectors    (amount → taps/strength)
    gaussian           baseline                                   (amount → sigma)
    fft_lowpass        per-channel FFT, attenuate highs           (amount → cutoff)
    edge_bilateral     Canny mask; bilateral off-edge, sharp on   (painterly)
    anisotropic        Perona–Malik via np.roll neighbours        (amount → iterations)
    luminance_guided   guide luminance sets blur strength/pixel   (guide required)

Guide clip (cross-video control). `guide` is an optional callable
`t -> frame | None`. When set, edge_bilateral and anisotropic take their edge /
structure information from the guide clip's frame at the same time index, and
luminance_guided becomes available. With no guide, the first two fall back to the
frame's own structure and luminance_guided is unavailable.
"""
from __future__ import annotations

from collections import deque
from typing import Callable, Iterable, Iterator

import cv2
import numpy as np

KINDS = ("smear", "temporal_average", "motion_flow", "gaussian", "fft_lowpass",
         "edge_bilateral", "anisotropic", "luminance_guided")
GUIDE_ONLY = ("luminance_guided",)


# ── individual operators ─────────────────────────────────────────────────────────
def _smear_kernel(angle_deg: float, length: int) -> np.ndarray:
    """Normalised line kernel at `angle_deg` (0 = horizontal, 90 = vertical)."""
    L = max(1, int(length))
    if L == 1:
        return np.ones((1, 1), np.float32)
    k = np.zeros((L, L), np.float32)
    c = (L - 1) / 2.0
    th = np.deg2rad(angle_deg)
    dx, dy = np.cos(th), -np.sin(th)
    ts = np.linspace(-c, c, L * 2)
    xs = np.clip(np.round(c + dx * ts).astype(int), 0, L - 1)
    ys = np.clip(np.round(c + dy * ts).astype(int), 0, L - 1)
    k[ys, xs] = 1.0
    total = k.sum()
    return k / total if total else k


def _smear(img, amount, angle):
    """Directional smear — the reference footage's blur is strongly anisotropic
    (measured ~7.4:1 horizontal:vertical gradient energy), which a gaussian
    cannot produce. One separable-ish line-kernel convolution, vectorised."""
    length = int(round(1 + amount * 47))
    if length <= 1:
        return img
    k = _smear_kernel(angle, length)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def _gaussian(img, amount):
    sigma = amount * 20.0
    if sigma <= 0.01:
        return img
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)


def _fft_lowpass(img, amount):
    cutoff = max(0.02, 1.0 - 0.95 * amount)
    if cutoff >= 0.999:
        return img
    h, w = img.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.ogrid[:h, :w]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    mask = (r <= cutoff).astype(np.float32)[..., None]
    spec = np.fft.fftshift(np.fft.fft2(img.astype(np.float32), axes=(0, 1)), axes=(0, 1))
    rec = np.fft.ifft2(np.fft.ifftshift(spec * mask, axes=(0, 1)), axes=(0, 1)).real
    return np.clip(rec, 0, 255).astype(np.uint8)


def _edge_bilateral(img, amount, structure):
    """Bilateral filter off-edge, original detail on-edge (painterly).
    `structure` supplies the edges — the guide clip's frame when one is set."""
    if amount <= 0.01:
        return img
    gray = cv2.cvtColor(structure, cv2.COLOR_RGB2GRAY)
    lo = int(40 + 60 * (1.0 - amount))
    edges = cv2.Canny(gray, lo, lo * 2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    d = int(3 + round(amount * 8)) | 1
    smooth = cv2.bilateralFilter(img, d, 40 + 120 * amount, 40 + 120 * amount)
    mask = (edges > 0)[..., None]
    return np.where(mask, img, smooth)


def _anisotropic(img, amount, structure):
    """Perona–Malik: conduction from the gradient (of the guide, when set), so
    edges are preserved while flat regions diffuse. Iterations capped."""
    iters = int(1 + round(amount * 11))
    if iters <= 0:
        return img
    f = img.astype(np.float32)
    g = structure.astype(np.float32)
    K = 20.0
    for _ in range(min(iters, 12)):
        for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
            dg = np.roll(g, shift, axis=axis) - g
            df = np.roll(f, shift, axis=axis) - f
            c = np.exp(-((dg / K) ** 2))          # conduction from structure
            f += 0.15 * c * df
        if structure is img:
            g = f
    return np.clip(f, 0, 255).astype(np.uint8)


def _luminance_guided(img, amount, guide, levels=5):
    """Guide luminance → per-pixel blur strength, quantised into `levels`
    precomputed blurs and indexed with a map. No per-pixel GaussianBlur calls."""
    max_sigma = max(0.5, amount * 20.0)
    lum = (0.299 * guide[..., 0] + 0.587 * guide[..., 1] + 0.114 * guide[..., 2])
    bucket = np.clip((lum / 256.0 * levels).astype(np.int64), 0, levels - 1)
    blurs = []
    for lv in range(levels):
        sigma = (lv / (levels - 1)) * max_sigma
        blurs.append(img if sigma <= 0.01
                     else cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma))
    stack = np.stack(blurs, axis=0)                       # (L,H,W,3)
    idx = np.broadcast_to(bucket[None, ..., None], (1,) + img.shape)
    return np.take_along_axis(stack, idx, axis=0)[0]


# ── streaming stage ──────────────────────────────────────────────────────────────
def stage(src: Iterable[np.ndarray], *, kind: str = "", amount: float = 0.0,
          guide: Callable[[int], np.ndarray | None] | None = None,
          levels: int = 5, angle: float = 90.0) -> Iterator[np.ndarray]:
    amount = float(amount)
    if not kind or kind not in KINDS or amount <= 0.0:
        yield from src
        return

    if kind == "temporal_average":
        window = max(1, 1 + int(round(amount * 23)))
        buf: deque[np.ndarray] = deque(maxlen=window)
        acc = None
        for frame in src:
            f = frame.astype(np.float32)
            if len(buf) == window and acc is not None:
                acc -= buf[0]
            buf.append(f)
            acc = f.copy() if acc is None else acc + f
            yield np.clip(acc / len(buf), 0, 255).astype(np.uint8)
        return

    if kind == "motion_flow":
        taps = max(1, int(round(1 + amount * 4)))
        strength = 0.5 + 3.5 * amount
        prev_gray = None
        grid = None
        for frame in src:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            if prev_gray is None:
                prev_gray = gray
                yield frame
                continue
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                0.5, 3, 15, 3, 5, 1.2, 0)
            prev_gray = gray
            h, w = frame.shape[:2]
            if grid is None or grid[0].shape != (h, w):
                grid = np.meshgrid(np.arange(w, dtype=np.float32),
                                   np.arange(h, dtype=np.float32))
            gx, gy = grid
            fx, fy = flow[..., 0] * strength, flow[..., 1] * strength
            acc = frame.astype(np.float32)
            count = 1.0
            for s in range(1, taps + 1):
                frac = s / float(taps)
                for sign in (-1.0, 1.0):
                    mx = (gx + sign * frac * fx).astype(np.float32)
                    my = (gy + sign * frac * fy).astype(np.float32)
                    acc += cv2.remap(frame, mx, my, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT).astype(np.float32)
                    count += 1.0
            yield np.clip(acc / count, 0, 255).astype(np.uint8)
        return

    # per-frame (optionally guide-driven) operators
    for t, frame in enumerate(src):
        g = guide(t) if guide is not None else None
        if kind == "smear":
            yield _smear(frame, amount, angle)
        elif kind == "gaussian":
            yield _gaussian(frame, amount)
        elif kind == "fft_lowpass":
            yield _fft_lowpass(frame, amount)
        elif kind == "edge_bilateral":
            yield _edge_bilateral(frame, amount, g if g is not None else frame)
        elif kind == "anisotropic":
            yield _anisotropic(frame, amount, g if g is not None else frame)
        elif kind == "luminance_guided":
            yield frame if g is None else _luminance_guided(frame, amount, g, levels)
        else:
            yield frame
