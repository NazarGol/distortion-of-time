"""
The sequencer — pool traversal as the compositional parameter.

This sits UPSTREAM of every effect. It decides, for a render, the ordered list of
pool stills to walk and how long each is held. Generators (weave, superimpose,
passthrough) consume that sequence instead of computing `t + something`
themselves, so order and rate become compositional controls rather than an
implicit "advance one frame per output frame".

Two axes the old design could not express:

  • RATE — `step_rate` is distinct stills per second, independent of output fps.
    Each still is held `output_fps / step_rate` output frames. (Reference
    footage: 12 stills/sec against 24fps output → each still held 2 frames.)
  • ORDER — the traversal need not be chronological, and need not be continuous.
    `jump_prob` / `jump_range` inject discontinuous leaps, so a stack of
    "neighbours" can hold genuinely unrelated moments and videos. That is the
    effect, not a bug.

The sequence spans the WHOLE selection, treated as one pool of stills — not one
clip walked in order.

`seed` is required and travels with the render (metadata + filename): this is for
an exhibited work, so an exact render must be reproducible.

Returns a `Sequence` rather than a bare list: the hold, seed and traversal
metadata have to travel with the steps for the generators and the readout.
`Sequence.steps` is the plain `list[FrameRef]`.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

import config

ORDERS = ("linear", "jitter", "drift", "shuffle", "pingpong", "crosscut", "similarity")

# similarity ordering is O(n²); cap how many stills take part so it stays usable
SIMILARITY_MAX = 1200


class FrameRef(NamedTuple):
    video_id: str
    frame_no: int


@dataclass
class Sequence:
    steps: list[FrameRef]          # distinct stills, in traversal order
    hold: float                    # output frames each still is held
    length: int                    # output frames
    order: str
    step_rate: float
    output_fps: float
    jump_prob: float
    jump_range: int
    seed: int
    neutral: bool = field(default=False)

    # ── mapping ──────────────────────────────────────────────────────────────
    def step_index(self, t: int) -> int:
        """Output frame → index into `steps` (this is where the hold happens)."""
        return int(t / self.hold) if self.hold != 1 else int(t)

    def ref(self, k: int) -> FrameRef:
        """Step index → pool frame, wrapping. Accepts out-of-range k so effects
        can reach traversal neighbours freely."""
        n = len(self.steps)
        return self.steps[int(k) % n] if n else FrameRef("", 0)

    def at(self, t: int) -> FrameRef:
        return self.ref(self.step_index(t))

    def describe(self) -> str:
        hold_txt = f" (hold {self.hold:g} frames)" if self.hold != 1 else ""
        bits = [f"{self.order} order",
                f"{self.step_rate:g} stills/sec{hold_txt}"]
        if self.jump_prob > 0:
            bits.append(f"jump {self.jump_prob:.2f}")
        bits.append(f"seed {self.seed}")
        return " · ".join(bits)


# ── the flat pool of stills ──────────────────────────────────────────────────────
def _flat_pool(selection: list[dict]) -> list[FrameRef]:
    flat: list[FrameRef] = []
    for v in selection:
        vid = v["video_id"]
        for i in range(max(1, int(v["n_frames"]))):
            flat.append(FrameRef(vid, i))
    return flat


# ── similarity ordering (visual nearest-neighbour chain) ─────────────────────────
def _descriptors(refs: list[FrameRef]) -> np.ndarray:
    """Tiny grayscale thumbnails as descriptors, cached on disk per video.

    Deliberately cheap (16×16 = 256-d). HOOK: swap this for CLIP/DINO embeddings
    to order by semantic rather than photometric similarity — everything
    downstream only needs an (N, D) float32 matrix.
    """
    by_video: dict[str, list[int]] = {}
    for idx, r in enumerate(refs):
        by_video.setdefault(r.video_id, []).append(idx)

    D = np.zeros((len(refs), 256), np.float32)
    for vid, idxs in by_video.items():
        cache_path = os.path.join(config.POOL_DIR, vid, "desc16.npy")
        table = None
        if os.path.isfile(cache_path):
            try:
                table = np.load(cache_path)
            except Exception:
                table = None
        if table is None:
            frames = sorted({refs[i].frame_no for i in idxs})
            n_max = max(frames) + 1 if frames else 0
            table = np.zeros((n_max, 256), np.float32)
            for fn in range(n_max):
                p = os.path.join(config.POOL_DIR, vid, f"{fn:06d}.jpg")
                img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                small = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
                table[fn] = small.astype(np.float32).ravel() / 255.0
            try:
                np.save(cache_path, table)
            except Exception:
                pass
        for i in idxs:
            fn = refs[i].frame_no
            if fn < len(table):
                D[i] = table[fn]
    # normalise so nearest-neighbour is a correlation
    D -= D.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(D, axis=1, keepdims=True)
    return D / np.maximum(norm, 1e-6)


def _similarity_order(flat: list[FrameRef], rng: np.random.Generator) -> list[int]:
    n = len(flat)
    if n <= 2:
        return list(range(n))
    # subsample if the pool is large, keeping the O(n²) chain tractable
    if n > SIMILARITY_MAX:
        pick = np.linspace(0, n - 1, SIMILARITY_MAX).astype(int)
    else:
        pick = np.arange(n)
    refs = [flat[i] for i in pick]
    D = _descriptors(refs)                      # (m, 256) unit rows
    m = len(refs)
    sim = D @ D.T                               # correlation matrix
    np.fill_diagonal(sim, -2.0)

    order_local = [0]
    unvisited = np.ones(m, bool)
    unvisited[0] = False
    cur = 0
    for _ in range(m - 1):
        row = np.where(unvisited, sim[cur], -2.0)
        nxt = int(np.argmax(row))
        order_local.append(nxt)
        unvisited[nxt] = False
        cur = nxt
    return [int(pick[i]) for i in order_local]


# ── base orderings (an ordering of indices into the flat pool) ───────────────────
def _base_order(order: str, flat: list[FrameRef], selection: list[dict],
                rng: np.random.Generator) -> list[int]:
    n = len(flat)
    if order == "shuffle":
        return list(rng.permutation(n))
    if order == "pingpong":
        fwd = list(range(n))
        return fwd + fwd[-2:0:-1] if n > 2 else fwd
    if order == "crosscut":
        # advance one step in time, switch source video every step
        starts, lens = [], []
        off = 0
        for v in selection:
            starts.append(off)
            L = max(1, int(v["n_frames"]))
            lens.append(L)
            off += L
        N = len(selection)
        out = []
        for s in range(n):
            k = s % N
            out.append(starts[k] + (s // N) % lens[k] if N else s)
        return out
    if order == "similarity":
        return _similarity_order(flat, rng)
    return list(range(n))                       # linear (also the base for jitter/drift)


# ── build ────────────────────────────────────────────────────────────────────────
def build_sequence(selection: list[dict], *, length: int, order: str = "linear",
                   step_rate: float | None = None, output_fps: float = 24.0,
                   jump_prob: float = 0.0, jump_range: int = 0,
                   seed: int = 0) -> Sequence:
    """Produce the ordered stills a render walks.

    length     — output frames
    step_rate  — distinct stills per second (default = output_fps, i.e. one still
                 per output frame: the neutral, current behaviour)
    """
    if not selection:
        raise ValueError("build_sequence needs at least one clip")
    output_fps = float(output_fps) or 24.0
    step_rate = float(step_rate) if step_rate else output_fps
    step_rate = max(0.05, step_rate)
    hold = output_fps / step_rate
    length = max(1, int(length))
    order = order if order in ORDERS else "linear"
    jump_prob = float(jump_prob or 0.0)
    jump_range = int(jump_range or 0)
    seed = int(seed)

    neutral = (order == "linear" and jump_prob <= 0.0 and abs(hold - 1.0) < 1e-9)

    flat = _flat_pool(selection)
    rng = np.random.default_rng(seed)
    bo = _base_order(order, flat, selection, rng)
    nb = len(bo)
    n_steps = max(1, int(np.ceil(length / hold)))

    # jitter window / drift step are derived from jump_range so the control
    # surface stays the one the spec defines
    window = max(1, jump_range) if jump_range else 8
    drift_k = max(1, (jump_range or 8) // 4)

    steps: list[FrameRef] = []
    cursor = -1
    for s in range(n_steps):
        if order == "jitter":
            cursor = s + int(rng.integers(-window, window + 1))
        elif order == "drift":
            cursor += int(rng.integers(-drift_k, drift_k + 1))
        else:
            cursor += 1
        if jump_prob > 0.0 and jump_range > 0 and rng.random() < jump_prob:
            cursor += int(rng.integers(-jump_range, jump_range + 1))
        steps.append(flat[bo[cursor % nb]])

    return Sequence(steps=steps, hold=hold, length=length, order=order,
                    step_rate=step_rate, output_fps=output_fps,
                    jump_prob=jump_prob, jump_range=jump_range, seed=seed,
                    neutral=neutral)


def seed_from(text: str) -> int:
    """Stable seed from arbitrary text (for 'random' buttons that stay reproducible)."""
    return int(hashlib.sha1(text.encode()).hexdigest()[:8], 16)
