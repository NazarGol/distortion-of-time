# Distortion of Time

A local tool for a programmatic video artwork about information / cognitive
warfare and the **distortion of time**. Degraded footage is re-woven as a
**vertical-strip time-displacement weave** — a slit-scan across multiple videos —
so a single output frame holds **many consecutive moments at once**.

> You do **not** need a Telegram account, a GPU, or any credentials to use this.
> The weave works on a plain folder of local video clips. The Telegram scraper is
> one *optional* way to fill that folder — never a dependency.

---

## The core effect — the weave mechanism

Given N source videos, all decoded to frame sequences and normalised to a common
resolution + fps:

- The output frame is divided into thin **vertical strips** (columns). Strip
  width in px is a parameter; small (1–3 px) gives the fine lenticular shimmer,
  wider gives coarser bands.
- Walking the strips **left → right** across one output frame, each successive
  strip **advances one step in time** and **rotates to the next source video**.
  For strip index `i` in output frame `t`:

  ```
  out[t] columns of strip i  =  the same columns taken from
      video[i mod N]  at  frame (t + i * frame_step)
  ```

  (`frame_step` default 1.) Column band `i` shows video `i mod N` at a later
  moment than band `i-1`: a single output image contains many consecutive
  moments woven across the N videos — **time runs across the width of the
  frame**. This is a *displacement* of image data in time, not a blend, mean,
  or overlay.
- For the next output frame (`t+1`) the base index advances by one, so the whole
  weave scrolls / flows as the video plays.
- Clips shorter than needed **loop** (sampling is modulo each clip's length).

Parameters: `strip_width` (px) · `frame_step` (frames per strip, may be
negative) · `orientation` (vertical / horizontal) · `source_mode`
(`rotate` = strip i ← video i mod N, default · `single` = pure slit-scan of one
clip). N is simply however many clips are loaded in the source pool.

Implementation: `weave.py`. Fully vectorised — for each source video, all of its
strips across all output frames are gathered in a single numpy fancy-index; no
per-pixel loops, no per-frame Python loops.

The old blur / feedback effects survive as **optional post-processing toggles**
applied after the weave, off by default.

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

## Using it — one screen

1. **Drop clips** into the file box (sample clips ship in `library/`, so you can
   render immediately). Every loaded clip appears in the **source pool**; the
   checked set, in order, is the N videos the weave rotates through.
2. Set the **weave controls**: strip width, frame step, orientation, source mode.
3. Optional **post toggles** (off by default): feedback trails (decaying echo of
   prior output) and one blur (temporal average / optical-flow / Gaussian /
   FFT low-pass / luminance-guided) with a single amount slider.
4. **Preview** renders a short, downscaled segment (fast). **Render** does the
   full length and gives you a download.

The small Telegram control (channel + max count) only activates when `.env`
credentials exist — see below.

---

## How it works (architecture)

```
config.py        paths + working-space defaults (incl. MAX_CLIP_SECONDS decode cap)
core.py          device detection (CPU / optional CUDA), RenderContext
frames/clip.py   decode video -> normalised (T,H,W,3) uint8 RGB stack; fps/size resample; cache
ingest/local.py  credential-free import of files into library/
ingest/telegram.py  optional wrapper around the Corpus Editor scraper
weave.py         THE CORE: strip time-displacement weave (slit-scan gather)
effects/         optional post filters: feedback trails + blur suite
render/writer.py frame stack -> H.264 mp4
app.py           single-screen Gradio UI (dark)
```

Long sources are truncated at `MAX_CLIP_SECONDS` (default 90 s) at decode time so
a stray full-length video can't eat all your RAM.

**Extending effects.** Add a subclass of `effects.base.Effect`, give it a `PARAMS`
schema, decorate it with `@register`, and import it in `effects/__init__.py`.

---

## GPU (optional)

CPU is the default and fully supported path. To use an NVIDIA GPU, install a
matching torch build — for an RTX 50-series (Blackwell) card:

```bash
./venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

CUDA is auto-detected at runtime and stays silent in the backend; everything
falls back to CPU when torch isn't present.

---

## Telegram scraping (optional)

If — and only if — you want to pull footage from public Telegram channels:

1. Get API credentials at <https://my.telegram.org> → *API development tools*.
2. Copy `.env.example` to `.env` and fill in `TG_API_ID` / `TG_API_HASH` / `TG_PHONE`.

Without a `.env`, the scrape control is simply disabled and the rest of the app
works normally. Downloaded videos land in `library/` like any other clip.

---

## Post-processing reference

| Effect | Key params |
| --- | --- |
| `feedback` | decay (persistence), mode (max/mean/add) |
| `temporal_average` | window (frames) |
| `motion_flow_blur` | taps, strength (Farneback optical flow) |
| `gaussian_blur` | sigma |
| `fft_lowpass` | cutoff (keep fraction) |
| `luminance_blur` | max_sigma, levels, invert |

In the UI each blur is driven by one 0–1 *amount* slider mapped onto its primary
parameter; the full parameter set is available programmatically via
`effects.build(name, params)`.
