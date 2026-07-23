"""
Render pipeline — pick ONE generator, then optionally apply post stages.

    generator (weave | superimpose | frames)
        → hyper-imposition (trails, optional)
        → blur (optional)
        → mp4, streamed frame by frame

Everything is an iterator: frames are produced, transformed and written one at a
time. Nothing here ever materialises the whole (T, H, W, 3) output, and the pool
is read on demand through the bounded LRU cache — so peak memory is flat in the
render length and in the corpus size.

`pool_weave` is left untouched; the weave generator delegates straight into it,
which is what keeps a neutral weave render byte-identical to previous output.
"""
from __future__ import annotations

import os
import time

import numpy as np

import config
import effects
import sequencer
from pool import FrameCache
from render.writer import stream_writer


def build_sequence_for(selection, params, *, num_frames=None):
    """The traversal every generator walks (see sequencer.py)."""
    fps = float(params.get("fps", config.FPS_FALLBACK))
    return sequencer.build_sequence(
        selection,
        length=output_len(selection, params, num_frames),
        order=params.get("order", "linear"),
        step_rate=params.get("step_rate") or None,
        output_fps=fps,
        jump_prob=float(params.get("jump_prob", 0.0) or 0.0),
        jump_range=int(params.get("jump_range", 0) or 0),
        seed=int(params.get("seed", 0) or 0),
    )


def build_stream(selection, cache, params, *, num_frames=None, sequence=None):
    """Compose generator + post stages into a single frame iterator."""
    mode = params.get("mode", "weave")
    gen = effects.GENERATORS.get(mode, effects.GENERATORS["weave"])
    fps = float(params.get("fps", config.FPS_FALLBACK))
    if sequence is None:
        sequence = build_sequence_for(selection, params, num_frames=num_frames)

    src = gen.generate(selection, cache, num_frames=num_frames, fps=fps,
                       sequence=sequence, **{
        k: v for k, v in params.items()
        if k in ("strip_width", "frame_step", "orientation", "source_mode",
                 "layers", "stride", "centred", "blend", "parallax", "amount",
                 "px_zoom", "px_pan", "px_rotate", "flow_source")
    })

    # ── optional guide clip (cross-video structure/luminance control) ──────────
    guide_id = params.get("guide_id") or ""
    guide_len = int(params.get("guide_len") or 0)
    guide = None
    if guide_id and guide_len > 0:
        def guide(t, _id=guide_id, _len=guide_len):
            return cache.get(_id, t % _len)

    # ── post stages ───────────────────────────────────────────────────────────
    persistence = float(params.get("persistence", 0.0) or 0.0)
    if persistence > 0:
        src = effects.feedback.stage(
            src, persistence=persistence, mode=params.get("trail_mode", "mean"),
            zoom=float(params.get("trail_zoom", 0.0) or 0.0),
            rotate=float(params.get("trail_rotate", 0.0) or 0.0),
            tx=float(params.get("trail_tx", 0.0) or 0.0),
            ty=float(params.get("trail_ty", 0.0) or 0.0))

    blur_kind = params.get("blur_kind") or ""
    blur_amount = float(params.get("blur_amount", 0.0) or 0.0)
    if blur_kind and blur_amount > 0:
        src = effects.blur.stage(src, kind=blur_kind, amount=blur_amount,
                                 guide=guide, levels=int(params.get("blur_levels", 5)),
                                 angle=float(params.get("blur_angle", 90.0)))
    return src


def output_len(selection, params, num_frames=None) -> int:
    mode = params.get("mode", "weave")
    if mode == "weave":
        return effects.weave.output_len(selection, params.get("source_mode", "rotate"),
                                        num_frames)
    if mode == "superimpose":
        return effects.superimpose.output_len(selection, num_frames)
    return effects.passthrough.output_len(selection, num_frames)


def render(selection, out_path, params, *, num_frames=None,
           cache: FrameCache | None = None, progress=None) -> dict:
    """Run the composed stream straight into an mp4. Returns metrics."""
    if not selection:
        raise ValueError("render needs at least one clip")
    cache = cache or FrameCache()
    fps = float(params.get("fps", config.FPS_FALLBACK))
    T_expected = output_len(selection, params, num_frames)

    seq = build_sequence_for(selection, params, num_frames=num_frames)
    stream = build_stream(selection, cache, params, num_frames=num_frames,
                          sequence=seq)
    t0 = time.time()
    writer = stream_writer(out_path, fps)
    written = 0
    try:
        for frame in stream:
            writer.append_data(np.ascontiguousarray(frame))
            written += 1
            if progress and written % 25 == 0:
                progress(written, T_expected)
    finally:
        writer.close()

    return {
        "path": out_path, "url": "/out/" + os.path.basename(out_path),
        "frames": written, "width": config.WORKING_WIDTH,
        "height": config.WORKING_HEIGHT, "sources": len(selection),
        "cache_hits": cache.hits, "cache_misses": cache.misses,
        "wall_s": time.time() - t0, "fps": fps,
        # the traversal travels with the render so it can be reproduced exactly
        "seed": seq.seed, "order": seq.order, "step_rate": seq.step_rate,
        "hold": seq.hold, "jump_prob": seq.jump_prob, "steps": len(seq.steps),
        "sequence": seq.describe(),
    }
