"""Sibling-effects tests. Uses ONLY material already in the pool — never generates clips."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pool, pool_weave, pipeline, effects, sequencer
from effects import depth as depthmod

vids = pool.list_videos()
assert vids, "pool is empty"
sel = [v for v in vids if v["n_frames"] > 100][:3] or vids[:3]
cache = pool.FrameCache()
T = 24

def frames_of(it): return [f.copy() for f in it]

# ── GROUND RULE: neutral weave == frozen pool_weave, bit for bit ──────────────
ref = frames_of(pool_weave.iter_frames(sel, cache, strip_width=2, frame_step=1.0,
        orientation="vertical", source_mode="rotate", num_frames=T))
got = frames_of(pipeline.build_stream(sel, cache, {
        "mode": "weave", "strip_width": 2, "frame_step": 1.0,
        "orientation": "vertical", "source_mode": "rotate", "fps": 24,
        # all new controls neutral:
        "persistence": 0.0, "blur_kind": "", "blur_amount": 0.0, "guide_id": ""},
        num_frames=T))
assert len(got) == len(ref), (len(got), len(ref))
assert all((a == b).all() for a, b in zip(got, ref)), "NEUTRAL WEAVE DIVERGED"
print(f"neutral weave bit-identical to pool_weave  ({len(ref)} frames) ✓")

# ── superimposition: sibling, no weave involved ───────────────────────────────
# layers are TRAVERSAL neighbours now, so a sequence is required (see sequencer.py)
one = [sel[0]]
def seq_for(material, n=8, **kw):
    return sequencer.build_sequence(material, length=n, output_fps=24, seed=1, **kw)

s1 = frames_of(effects.superimpose.generate(one, cache, layers=8, stride=5, fps=24,
        sequence=seq_for(one)))
assert s1[0].shape == (360,640,3) and s1[0].dtype == np.uint8
s2 = frames_of(effects.superimpose.generate(sel, cache, layers=len(sel), stride=1, fps=24,
        sequence=seq_for(sel, order="crosscut")))
assert not np.array_equal(s1[0], s2[0]), "stride/selection made no difference"
print(f"superimpose: 1-clip stack (8 layers, stride 5) and {len(sel)}-clip crosscut both render ✓")

# blends
for b in effects.superimpose.BLENDS:
    f = frames_of(effects.superimpose.generate(sel, cache, layers=3, blend=b, fps=24,
            sequence=seq_for(sel, n=2)))
    assert f[0].dtype == np.uint8
print("blends:", ", ".join(effects.superimpose.BLENDS), "✓")

# ── parallax modes ────────────────────────────────────────────────────────────
for mode in ("none", "affine", "flow"):
    f = frames_of(effects.superimpose.generate(one, cache, layers=4, stride=2,
            parallax=mode, amount=0.6, fps=24, sequence=seq_for(one, n=3)))
    assert f[0].dtype == np.uint8
    print(f"parallax {mode}: ok ✓")
ok, why = depthmod.available()
print(f"parallax depth: available={ok} — {why}")
f = frames_of(effects.superimpose.generate(one, cache, layers=3, parallax="depth",
        amount=0.6, fps=24, sequence=seq_for(one, n=2)))
assert f[0].dtype == np.uint8
print("parallax depth: falls back cleanly when dependency absent ✓")

# ── hyper-imposition with transform ───────────────────────────────────────────
src = effects.passthrough.generate(one, cache, sequence=seq_for(one, n=10))
fb = frames_of(effects.feedback.stage(src, persistence=0.8, mode="max", zoom=0.02, rotate=0.5))
assert len(fb) == 10 and fb[0].dtype == np.uint8
print("hyper-imposition (persistence .8, zoom .02, rotate .5°): ok ✓")

# ── every blur, on PLAIN FRAMES (no weave, no superimposition) ────────────────
guide_v = sel[1] if len(sel) > 1 else sel[0]
guide = lambda t, v=guide_v: cache.get(v["video_id"], t % v["n_frames"])
for kind in effects.blur.KINDS:
    src = effects.passthrough.generate(one, cache, sequence=seq_for(one, n=6))
    out = frames_of(effects.blur.stage(src, kind=kind, amount=0.5, guide=guide))
    assert len(out) == 6 and out[0].shape == (360,640,3), kind
    print(f"blur {kind}: ok on plain frames ✓")

# guide-less fallback for the guide-driven ones
for kind in ("edge_bilateral", "anisotropic"):
    src = effects.passthrough.generate(one, cache, sequence=seq_for(one, n=3))
    out = frames_of(effects.blur.stage(src, kind=kind, amount=0.5, guide=None))
    assert len(out) == 3
print("edge_bilateral + anisotropic fall back to own structure with no guide ✓")

print("\nALL SIBLING-EFFECT TESTS PASS")
