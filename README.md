# Distortion of Time

A local tool for a programmatic video artwork about information / cognitive
warfare and the **distortion of time**. Footage is decomposed into a **pool of
frames on disk**, then re-woven as a **vertical-strip time-displacement weave** —
a slit-scan across the pool — so a single output frame holds **many consecutive
moments at once**.

> You do **not** need a Telegram account, a GPU, or any credentials. The tool
> works on a plain folder of local video clips; the Telegram scraper is one
> *optional* way to fill it.

---

## The core effect — the weave mechanism

Given the frame pool, the output frame is divided into thin **vertical strips**
(columns). Walking the strips **left → right**, each successive strip **advances
in time** and (in corpus mode) **rotates to the next source video**:

```
out[t] columns of strip i  =  the same columns taken from
    video[i mod N]  at  frame (t + i * frame_step)
```

Column band `i` shows a later moment than band `i-1`: **time runs across the
width of the frame**. This is a *displacement* of image data in time — not a
blend, mean, or overlay. The maths live in `weave.py` and are frozen + unit
verified; `pool_weave.py` reproduces them bit-for-bit while streaming from disk
(`tests/test_equivalence.py` asserts the two match exactly).

Two selection modes: **one clip** (strips walk consecutive frames of a single
video — a pure slit-scan) and **whole corpus** (strips rotate across the pool).
`N` is not a setting — it is however much material is in the pool.

---

## Architecture — a frame pool, streamed

```
scrape / import  →  decompose each video into frames on disk  →  frame pool
                 →  weave stitches strips out of the pool  →  streamed mp4
```

- **Decompose** (`pool.py` + `preprocess.py`). Every video is decomposed **once**
  into individual working-resolution JPEG frames at `pool/<video_id>/000123.jpg`.
  A **SQLite index** (`pool/index.db`) records every frame (frame_id, video_id,
  frame_number, timestamp, path) and each video's metadata. Decomposition is
  **idempotent** — a video already in the index is never redone.
- **Preprocessing at decompose time** fixes the black bands and incoherence:
  - **Letterbox / pillarbox bars** are auto-detected per video and cropped, so no
    strip is ever sampled inside a black bar.
  - Frames are **centre-cropped to the common aspect** (never squashed) and each
    video's **brightness/contrast is normalised** toward a shared target so
    adjacent strips from different sources don't clash. Exposure match is a
    decompose toggle, **on by default**.
- **The weave reads on demand.** For each output frame, each strip needs exactly
  one column band from one pool frame; those frames are pulled through a bounded
  LRU cache (`pool.FrameCache`). **Memory stays flat regardless of corpus size.**
- **The output is streamed** frame-by-frame straight into the mp4 writer — the
  full `(T, H, W, 3)` stack is never held in RAM.

```
config.py        paths, working resolution, fps fallback, cache size, thresholds
preprocess.py    bar detection + crop, aspect crop, exposure normalisation
pool.py          decompose + SQLite index + bounded LRU frame cache + thumbnails
weave.py         THE CORE maths (frozen, verified)
pool_weave.py    streaming, pool-backed renderer (bit-identical to weave.py)
effects/         optional post filters (feedback trails, blur) — off, out of the way
render/writer.py streaming H.264 mp4 writer
server.py        FastAPI: static frontend + JSON API over pool & renderer
web/index.html   the single-page instrument UI (no Gradio)
app.py           launcher (uvicorn)
```

---

## Quick start

```bash
cd distortion_of_time
./run.sh                 # creates the venv + installs deps on first run, then launches
```

Then open **http://127.0.0.1:8000**.

Requirements: Python 3.10+ (tested on 3.13). Everything runs on CPU.
`imageio-ffmpeg` bundles its own ffmpeg — no system ffmpeg needed.

---

## Using it — one screen

- **Preview** dominates: it plays the woven output, with a thin scrub bar below.
- **Four controls**, one row:
  - **strip width** (px) — 1–3 px = fine lenticular shimmer; wider = bands.
  - **time span** (seconds) — *how much time the frame holds across its width*.
    `frame_step` is derived internally as `span × fps ÷ n_strips`, so the total
    displacement across the frame equals `span` seconds. (Default ~1.5 s. Setting
    `frame_step = 1` at 2 px strips would make the frame span ~27 s — never the
    default.)
  - **direction** — forward / back (time flows either way across the frame) and
    vertical / horizontal strips.
  - **draw from** — one clip / whole corpus.
- A **live readout** under the controls: `320 strips · 0.11 frames per strip ·
  frame holds 1.5s of time`.
- The **corpus** is a filmstrip of thumbnails with a `+N` overflow; click a
  thumbnail to pick the one-clip source. Import / decompose / scrape live behind
  the **corpus** button.
- One **render** button.

Working **fps = each source clip's own fps** (probed at decompose; falls back to
24). We never silently downsample — each strip advances one *frame*, so fps is the
effect's temporal resolution.

Aesthetic: near-black, monospace, an instrument rather than a control panel.

---

## Telegram scraping (optional)

1. Get API credentials at <https://my.telegram.org> → *API development tools*.
2. Copy `.env.example` to `.env` and fill in `TG_API_ID` / `TG_API_HASH` / `TG_PHONE`.

Without a `.env`, the scrape control in the corpus panel is simply disabled and
everything else works. Scraped videos land in `library/` and are decomposed into
the pool like any other clip.

---

## GPU (optional)

CPU is the default and fully supported path. To use an NVIDIA GPU, install a
matching torch build (e.g. RTX 50-series / Blackwell):

```bash
./venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

CUDA is auto-detected at runtime and stays silent; everything falls back to CPU.

---

## Post-processing (optional, out of the way)

`feedback` (decaying trails) and a blur suite (`temporal_average`,
`motion_flow_blur`, `gaussian_blur`, `fft_lowpass`, `luminance_blur`) live in
`effects/` and can be applied after the weave via `pool_weave.render_weave(...,
post=…)`. They are off by default and deliberately absent from the four-control
main screen.
