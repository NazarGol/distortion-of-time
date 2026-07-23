"""
Streaming, pool-backed renderer for the strip time-displacement weave.

The maths are exactly those of weave.py (verified) — this module does NOT change
them. It only changes what feeds the weave and how the output is written:

  • Material is read from the frame pool on demand. For each output frame, each
    strip needs exactly ONE column band from ONE pool frame; we fetch only those
    frames through a bounded LRU cache and copy the band into the output.
  • The output is streamed frame-by-frame straight into the mp4 writer — the full
    (T, H, W, 3) stack is never materialised.

Per (output frame t, strip i):

    video index v  = i mod N          (rotate)      | 0 (single)
    displacement   = round(i * frame_step)          # integer step ⇒ i*step, == weave.py
    source frame   = (t + displacement) mod n_frames[v]
    output band    = columns [i*sw : (i+1)*sw]       (rows, if horizontal)

`iter_frames` is the pure gather (no encoding); tests/test_equivalence.py drives
it with an in-memory cache and asserts it reproduces weave.weave() bit-for-bit.
"""
from __future__ import annotations

import time
from typing import Callable, Iterator, Protocol

import numpy as np

import config
from pool import FrameCache
from render.writer import stream_writer


class _Cache(Protocol):
    def get(self, video_id: str, frame_number: int) -> np.ndarray: ...


def _plan_strips(axis_len: int, sw: int, n: int, frame_step: float, source_mode: str):
    """Precompute (band_start, band_end, video_index, displacement) per strip."""
    n_strips = (axis_len + sw - 1) // sw
    strips = []
    for i in range(n_strips):
        a = i * sw
        b = min(a + sw, axis_len)
        v = (i % n) if source_mode == "rotate" else 0
        disp = int(round(i * frame_step))
        strips.append((a, b, v, disp))
    return strips


def _output_len(videos, source_mode, num_frames) -> int:
    if num_frames:
        return max(1, int(num_frames))
    lens = [max(1, int(v["n_frames"])) for v in videos]
    if source_mode != "rotate" or len(videos) == 1:
        return lens[0]
    return max(lens)


def iter_frames(
    videos: list[dict],
    cache: _Cache,
    *,
    strip_width: int = 2,
    frame_step: float = 1.0,
    orientation: str = "vertical",
    source_mode: str = "rotate",
    num_frames: int | None = None,
    width: int = config.WORKING_WIDTH,
    height: int = config.WORKING_HEIGHT,
) -> Iterator[np.ndarray]:
    """Yield woven output frames one at a time. Pure gather — no encoding, no
    buffering. Each strip pulls one column band from one pool frame via `cache`."""
    if not videos:
        raise ValueError("iter_frames needs at least one video")
    W, H = int(width), int(height)
    sw = max(1, int(strip_width))
    vertical = orientation != "horizontal"
    axis_len = W if vertical else H
    lens = [max(1, int(v["n_frames"])) for v in videos]
    strips = _plan_strips(axis_len, sw, len(videos), frame_step, source_mode)
    T = _output_len(videos, source_mode, num_frames)

    for t in range(T):
        out = np.empty((H, W, 3), np.uint8)
        for a, b, v, disp in strips:
            fn = (t + disp) % lens[v]
            src = cache.get(videos[v]["video_id"], fn)
            if vertical:
                out[:, a:b] = src[:, a:b]
            else:
                out[a:b, :] = src[a:b, :]
        yield out


def render_weave(
    videos: list[dict],
    out_path: str,
    *,
    strip_width: int = 2,
    frame_step: float = 1.0,
    orientation: str = "vertical",
    source_mode: str = "rotate",
    fps: float = config.FPS_FALLBACK,
    num_frames: int | None = None,
    cache: FrameCache | None = None,
    post: Callable[[np.ndarray], np.ndarray] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> dict:
    """Render the weave from pool `videos` straight to `out_path` (streamed).
    Returns render metrics. `post` (feedback/blur) is off by default and, when
    given, buffers frames since those effects need temporal neighbours."""
    if not videos:
        raise ValueError("render_weave needs at least one pool video")
    cache = cache or FrameCache()
    T_total = _output_len(videos, source_mode, num_frames)
    frames_iter = iter_frames(
        videos, cache, strip_width=strip_width, frame_step=frame_step,
        orientation=orientation, source_mode=source_mode, num_frames=num_frames)

    t0 = time.time()
    n_strips = (
        (config.WORKING_WIDTH if orientation != "horizontal" else config.WORKING_HEIGHT)
        + max(1, int(strip_width)) - 1) // max(1, int(strip_width))

    if post is None:                               # streaming fast path — flat memory
        writer = stream_writer(out_path, fps)
        written = 0
        try:
            for t, out in enumerate(frames_iter):
                writer.append_data(out)
                written += 1
                if progress and t % 10 == 0:
                    progress(t + 1, T_total)
        finally:
            writer.close()
        T = written
    else:                                          # post-effects path (buffered)
        buf = list(frames_iter)
        frames = post(np.stack(buf, axis=0))
        writer = stream_writer(out_path, fps)
        try:
            for f in frames:
                writer.append_data(np.ascontiguousarray(f))
        finally:
            writer.close()
        T = len(frames)

    return {
        "path": out_path, "frames": T,
        "width": config.WORKING_WIDTH, "height": config.WORKING_HEIGHT,
        "n_strips": n_strips, "sources": len(videos),
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "wall_s": time.time() - t0,
    }
