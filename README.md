# Distortion of Time

A local tool for a programmatic video artwork about information / cognitive
warfare and the **distortion of time**. Degraded footage is broken into frames
and re-woven — sliced into interlaced strips, time-shifted, and superimposed — so
a single output frame holds **several moments at once**.

> You do **not** need a Telegram account, a GPU, or any credentials to use this.
> The compositor works on a plain folder of local video clips. The Telegram
> scraper is one *optional* way to fill that folder — never a dependency.

---

## Quick start

```bash
cd distortion_of_time
./run.sh                 # creates the venv + installs deps on first run, then launches
```

or manually:

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python app.py
```

Then open the printed local URL (default http://127.0.0.1:7860).

Requirements: Python 3.10+ (tested on 3.13). Everything runs on CPU. `imageio-ffmpeg`
bundles its own ffmpeg, so no system ffmpeg install is needed.

---

## Using it

1. **Import / Scrape tab** — drag video files in and click *Import*, or paste a
   folder path to import every clip in it. (Some sample clips ship in `library/`
   already, so you can render immediately.)
2. **Library tab** — thumbnails of everything currently available.
3. **Compose tab** — an ordered effect chain: **Combine → Feedback → Blur**.
   - **Layers.** Enable 2+ layers, pick a clip each, and set a **temporal offset**
     (frames): the layer shows the moment `t − offset`, wrapping around the clip.
     **Use the same clip in two layers with different offsets** to turn one clip
     into many moments — that's the whole idea. Each layer also has scale + opacity.
   - **1 · Combine** — `interlace` (the signature weave: strip width 1–3 px = fine
     lenticular shimmer, wider = bands; orientation; scroll speed for
     parallax-in-time; feather to soften seams) **or** `superimpose` (blend N layers
     with a mode: normal / add / screen / multiply / difference + mix) **or** none.
   - **2 · Feedback trails** (hyper-imposition) — blend each frame into a decaying
     accumulator of prior output for echo/trails. *Persistence* + mode
     (max / mean / add).
   - **3 · Blur** — temporal average (long exposure), motion-compensated optical
     flow, Gaussian, frequency-domain low-pass, or luminance-guided.
   - **Preview** renders a short, downscaled segment (fast). **Render** does the
     full clip and gives you a download.
   - **Presets** — *Save* the current composition to JSON, or *Load & render* a
     saved one (reproduce a render / share a look with a collaborator). Three
     starter presets ship in `compositions/`.

---

## How it works (architecture)

Ingest and effects are **decoupled** — the compositor never imports the scraper.

```
config.py        paths + working-space defaults
core.py          device detection (CPU / optional CUDA), RenderContext
frames/clip.py   decode video -> normalised (N,H,W,3) uint8 RGB stack; fps/size resample; cache
ingest/local.py  credential-free import of files / folders into library/
effects/         Effect base class + registry; interlace.py (the signature weave)
composition.py   Composition = N time-offset layers + an ordered effect chain -> frames
render/writer.py frame stack -> H.264 mp4
app.py           Gradio UI
```

**Data model.** A `Composition` holds N source **layers** (clip + temporal offset +
scale) and an ordered **effect chain**. Each effect transforms a *list of layers*
(each layer an `(T,H,W,3)` array); *combiner* effects like interlace collapse the
layers into one. Everything is vectorised with numpy/OpenCV — no per-pixel Python
loops.

**Extending effects.** Add a subclass of `effects.base.Effect`, give it a `PARAMS`
schema, decorate it with `@register`, and import it in `effects/__init__.py`. It
becomes available to any composition. (Superimposition and the blur suite plug in
here.)

---

## GPU (optional)

CPU is the default and fully supported path. To use an NVIDIA GPU, install a
matching torch build — for an RTX 50-series (Blackwell) card:

```bash
./venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

The tool auto-detects `torch` + CUDA at runtime (shown as the *Compute* line in the
UI) and falls back to CPU when it isn't present.

---

## Telegram scraping (optional)

If — and only if — you want to pull footage from public Telegram channels:

1. Get API credentials at <https://my.telegram.org> → *API development tools*.
2. Copy `.env.example` to `.env` and fill in `TG_API_ID` / `TG_API_HASH` / `TG_PHONE`.

Without a `.env`, the scrape tab simply shows a friendly notice and the rest of the
app works normally. Downloaded videos land in `library/` like any other clip.

---

## Effects reference

| Effect | Kind | Key params |
| --- | --- | --- |
| `interlace` | combiner | strip_width, orientation, phase_speed, feather |
| `superimpose` | combiner | blend (normal/add/screen/multiply/difference), mix; per-layer opacity |
| `feedback` | filter | decay (persistence), mode (max/mean/add) |
| `temporal_average` | filter | window (frames) |
| `motion_flow_blur` | filter | taps, strength (Farneback optical flow) |
| `gaussian_blur` | filter | sigma |
| `fft_lowpass` | filter | cutoff (keep fraction) |
| `luminance_blur` | filter | max_sigma, levels, invert |

## Build status

- [x] **Phase 0** — scaffold, deps, app launches
- [x] **Phase 1** — local import, frame extraction, render passthrough
- [x] **Phase 2** — interlacing / lenticular effect, Preview + Render
- [x] **Phase 3** — superimposition + hyper-imposition (feedback trails)
- [x] **Phase 4** — blur suite (temporal averaging, optical-flow, Gaussian, FFT, luminance-guided)
- [x] **Phase 5** — Telegram scraper wired into the Import tab (credential-gated)
- [x] **Phase 6** — save / load presets in the UI; starter presets in `compositions/`

**Left as extension points** (documented in `effects/blur.py`): edge-guided
bilateral, anisotropic diffusion, depth-based, dynamic-kernel and full
spatio-temporal 3-D blurs, plus cross-source blur variants (run *before* the
combine stage where 2+ layers still exist). Add an `Effect` subclass, decorate it
`@register`, import it in `effects/__init__.py` — it wires itself into the engine.
