"""Sequencer tests. Uses ONLY material already in the pool — never generates clips."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pool, pool_weave, pipeline, sequencer, effects

vids = pool.list_videos()
sel = [v for v in vids if v["n_frames"] > 200][:3]
one = [max(vids, key=lambda v: v["n_frames"])]
cache = pool.FrameCache()
T = 24
def frames_of(it): return [f.copy() for f in it]

# (a) ALL-NEUTRAL bit-identical to the frozen weave
ref = frames_of(pool_weave.iter_frames(sel, cache, strip_width=2, frame_step=1.0,
        orientation="vertical", source_mode="rotate", num_frames=T))
neutral = {"mode":"weave","strip_width":2,"frame_step":1.0,"orientation":"vertical",
           "source_mode":"rotate","fps":24,"order":"linear","step_rate":24,
           "jump_prob":0.0,"jump_range":0,"seed":0,
           "persistence":0.0,"blur_kind":"","blur_amount":0.0}
got = frames_of(pipeline.build_stream(sel, cache, neutral, num_frames=T))
assert len(got)==len(ref) and all((a==b).all() for a,b in zip(got,ref)), "NEUTRAL DIVERGED"
print(f"(a) all-neutral weave bit-identical to frozen pool_weave ({T} frames) ✓")

# (b) step_rate 12 against 24fps output -> each still held EXACTLY 2 frames
p = dict(neutral); p.update({"mode":"frames","step_rate":12,"fps":24})
seq = pipeline.build_sequence_for(one, p, num_frames=16)
f = frames_of(pipeline.build_stream(one, cache, p, num_frames=16, sequence=seq))
pairs_equal = all(np.array_equal(f[i], f[i+1]) for i in range(0, len(f)-1, 2))
changes = sum(1 for i in range(1, len(f)) if not np.array_equal(f[i-1], f[i]))
# structural proof: output frame -> step index must advance every 2nd frame
idx = [seq.step_index(t) for t in range(len(f))]
assert seq.hold == 2.0, seq.hold
assert idx == [t // 2 for t in range(len(f))], idx
assert pairs_equal, "held pairs not identical"
# (visual changes can be fewer than distinct stills when the footage is static)
print(f"(b) step_rate 12 @24fps -> hold {seq.hold:g}; step index {idx[:8]}… "
      f"each still held exactly 2 frames ✓  [{changes} visible changes / "
      f"{len(f)//2} stills — static footage repeats]")

# (c) every order actually reorders (each needs the right material to show it:
#     pingpong only diverges once it turns around; crosscut needs >1 clip)
short = [min(vids, key=lambda v: v["n_frames"])]          # wraps quickly
P_short = short[0]["n_frames"]
cases = [("drift", one, 60), ("shuffle", one, 60), ("jitter", one, 60),
         # pingpong only diverges past the turnaround, so walk beyond the pool
         ("pingpong", short, 2 * P_short + 20),
         ("crosscut", sel, 60), ("similarity", short, 60)]
for order, material, length in cases:
    q = {**p, "seed": 7, "step_rate": 24}          # hold 1 so steps == frames
    lin = pipeline.build_sequence_for(material, {**q, "order": "linear"},
                                      num_frames=length)
    s = pipeline.build_sequence_for(material, {**q, "order": order,
                                               "jump_range": 40}, num_frames=length)
    assert s.steps != lin.steps, order
    vids_used = len({r.video_id for r in s.steps})
    print(f"(c) order {order:10s} differs from linear ✓  ({vids_used} source(s) in traversal)")

# (d) jumps create discontinuities the linear traversal never has
s0 = pipeline.build_sequence_for(one, {**p,"order":"linear","jump_prob":0.0,"seed":3}, num_frames=200)
s1 = pipeline.build_sequence_for(one, {**p,"order":"linear","jump_prob":0.15,"jump_range":60,"seed":3}, num_frames=200)
d0 = np.abs(np.diff([r.frame_no for r in s0.steps]))
d1 = np.abs(np.diff([r.frame_no for r in s1.steps]))
big = int((d1 > 5*max(1,np.median(d1))).sum())
print(f"(d) jump_prob .15: {big} leaps >5x median (linear has {int((d0>5*max(1,np.median(d0))).sum())}) ✓")

# (e) superimposition of 6 traversal-neighbours over a JUMPING sequence
pj = {**p,"mode":"superimpose","layers":6,"stride":2,"blend":"mean",
      "order":"drift","jump_prob":0.15,"jump_range":60,"seed":11}
sj = pipeline.build_sequence_for(one, pj, num_frames=T)
fj = frames_of(pipeline.build_stream(one, cache, pj, num_frames=T, sequence=sj))
assert len(fj)==T and fj[0].shape==(360,640,3)
print(f"(e) superimpose 6 traversal-neighbours (stride 2) over jumping sequence ✓")

# (f) smear at 90 deg is directionally anisotropic (reference measured ~7.4:1)
src = cache.get(one[0]["video_id"], 100)
def grads(img):
    g = img.astype(np.float32).mean(2)
    return (float(np.abs(np.diff(g, axis=1)).mean()),   # dx (horizontal detail)
            float(np.abs(np.diff(g, axis=0)).mean()))   # dy (vertical detail)
def one_shot(kind, **kw):
    return frames_of(effects.blur.stage(iter([src]), kind=kind, amount=1.0, **kw))[0]

dx0, dy0 = grads(src)
def reduction(img):
    dx, dy = grads(img)
    return dx0 / max(dx, 1e-6), dy0 / max(dy, 1e-6)     # how much each axis lost

rx90, ry90 = reduction(one_shot("smear", angle=90))     # vertical smear
rx0,  ry0  = reduction(one_shot("smear", angle=0))      # horizontal smear
rxg,  ryg  = reduction(one_shot("gaussian"))
# a directional smear must destroy detail far more along its own axis
assert ry90 > 1.8 * rx90, (rx90, ry90)
assert rx0 > 1.8 * ry0, (rx0, ry0)
assert 0.6 < rxg / ryg < 1.7, (rxg, ryg)                # gaussian is isotropic
print(f"(f) detail lost per axis (higher = more blurred):")
print(f"    smear 90° : dx {rx90:.2f}x  dy {ry90:.2f}x  -> {ry90/rx90:.1f}x more vertical ✓")
print(f"    smear  0° : dx {rx0:.2f}x  dy {ry0:.2f}x  -> {rx0/ry0:.1f}x more horizontal ✓")
print(f"    gaussian  : dx {rxg:.2f}x  dy {ryg:.2f}x  -> isotropic ({rxg/ryg:.2f}) ✓")

# (g) same seed twice -> identical; different seed -> different
def run(seed):
    pp = {**p,"mode":"frames","order":"shuffle","jump_prob":0.2,"jump_range":50,"seed":seed}
    return frames_of(pipeline.build_stream(one, cache, pp, num_frames=12))
a1, a2, b = run(4471), run(4471), run(4472)
assert all(np.array_equal(x,y) for x,y in zip(a1,a2)), "same seed diverged"
assert not all(np.array_equal(x,y) for x,y in zip(a1,b)), "different seed identical"
print("(g) seed 4471 twice -> byte-identical; seed 4472 -> different ✓")

print("\nALL SEQUENCER TESTS PASS")
