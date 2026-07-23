"""
Distortion of Time — single-screen UI for the strip time-displacement weave.

Drop clips → the loaded set is the source pool → weave controls (strip width,
frame step, orientation, source mode) → Preview (short, downscaled) → Render →
mp4 download. Optional post toggles (feedback trails, one blur), off by default.
Local import is credential-free; the Telegram scrape control is unobtrusive and
only lights up when .env credentials exist. CUDA is auto-detected in the backend.

Run:  python app.py    (inside the project venv)
"""
from __future__ import annotations

import os
import time

import gradio as gr

import config
import effects
from core import RenderContext, get_device
from frames import clip as clipmod
from ingest import local as ingest_local
from ingest import telegram as ingest_tg
from render.writer import write_mp4
from weave import weave

BLUR_CHOICES = ["(none)", "temporal_average", "motion_flow_blur",
                "gaussian_blur", "fft_lowpass", "luminance_blur"]

_FORCE_DARK_JS = """
() => {
  const url = new URL(window.location);
  if (url.searchParams.get('__theme') !== 'dark') {
    url.searchParams.set('__theme', 'dark');
    window.location.href = url.href;
  }
}
"""

_CSS = """
.gradio-container { max-width: 1100px !important; }
footer { display: none !important; }
"""


# ── helpers ──────────────────────────────────────────────────────────────────────
def _pool_choices() -> list[str]:
    return ingest_local.library_names()


def _default_pool() -> list[str]:
    names = _pool_choices()
    samples = [n for n in names if n.startswith("sample_")]
    return samples or names


def _even(x: int) -> int:
    return max(2, int(x) - (int(x) % 2))


def _load_pool(names: list[str], w: int, h: int, fps: int):
    clips = []
    for n in names or []:
        path = os.path.join(config.LIBRARY_DIR, n)
        if os.path.isfile(path):
            clips.append(clipmod.extract(path, width=w, height=h, fps=fps))
    return clips


def _blur_params(blur_type: str, amount: float) -> dict:
    """One 0–1 'amount' slider drives each blur's primary strength."""
    a = float(amount)
    return {
        "temporal_average": {"window": 1 + round(a * 23)},
        "motion_flow_blur": {"taps": max(1, round(1 + a * 5)),
                             "strength": 0.2 + 3.8 * a},
        "gaussian_blur": {"sigma": a * 20.0},
        "fft_lowpass": {"cutoff": max(0.02, 1.0 - 0.95 * a)},
        "luminance_blur": {"max_sigma": 1.0 + 23.0 * a, "levels": 5},
    }.get(blur_type, {})


def _run(pool, sw, step, orient, mode, fb_on, fb_decay, blur_type, blur_amt,
         width, height, fps, is_preview: bool):
    if not pool:
        raise gr.Error("Load / select at least one clip for the source pool.")
    w, h = _even(width), _even(height)
    if is_preview and w > config.PREVIEW_MAX_WIDTH:
        scale = config.PREVIEW_MAX_WIDTH / float(w)
        w, h = _even(config.PREVIEW_MAX_WIDTH), _even(round(h * scale))

    t0 = time.time()
    clips = _load_pool(pool, w, h, int(fps))
    if not clips:
        raise gr.Error("None of the selected clips could be decoded.")

    num_frames = config.PREVIEW_MAX_FRAMES if is_preview else None
    frames = weave(clips, strip_width=int(sw), frame_step=int(step),
                   orientation=orient, source_mode=mode, num_frames=num_frames)

    # optional post-processing, applied AFTER the weave
    ctx = RenderContext(fps=int(fps), width=w, height=h,
                        is_preview=is_preview, device=get_device())
    post = []
    if fb_on:
        frames = effects.build("feedback", {"decay": float(fb_decay)}).apply([frames], ctx)[0]
        post.append("feedback")
    if blur_type != "(none)" and blur_amt > 0:
        frames = effects.build(blur_type, _blur_params(blur_type, blur_amt)).apply([frames], ctx)[0]
        post.append(blur_type)

    prefix = "preview" if is_preview else "render"
    out = os.path.join(config.OUTPUT_DIR, f"{prefix}_{int(time.time())}.mp4")
    write_mp4(frames, out, fps=int(fps))
    desc = (f"{len(frames)}f @ {w}×{h} · {len(clips)} sources · strips {int(sw)}px · "
            f"step {int(step)}" + (f" · post: {'+'.join(post)}" if post else "")
            + f" · {time.time()-t0:.1f}s")
    return out, desc


def do_preview(*args):
    out, desc = _run(*args, is_preview=True)
    return out, f"Preview · {desc}"


def do_render(*args):
    out, desc = _run(*args, is_preview=False)
    return out, out, f"Rendered · {desc}\n{out}"


# ── ingest callbacks ─────────────────────────────────────────────────────────────
def do_upload(files, current_sel):
    paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in (files or [])]
    imported = ingest_local.import_files([p for p in paths if p])
    sel = list(current_sel or []) + imported
    return (gr.update(choices=_pool_choices(), value=sel),
            f"Imported {len(imported)} clip(s)." if imported else "No videos found in upload.")


def do_scrape(channel, count, current_sel):
    msg = ingest_tg.scrape(channel, int(count))
    return gr.update(choices=_pool_choices(), value=list(current_sel or [])), msg


# ── UI ───────────────────────────────────────────────────────────────────────────
def build_ui():
    tg_ok, tg_msg = ingest_tg.availability()

    with gr.Blocks(title="Distortion of Time", theme=gr.themes.Monochrome(),
                   css=_CSS, js=_FORCE_DARK_JS) as demo:
        gr.Markdown("## Distortion of Time — strip time-displacement weave")

        with gr.Row():
            up = gr.File(label="Drop clips here", file_count="multiple",
                         file_types=["video"], height=110, scale=2)
            with gr.Column(scale=1):
                tg_channel = gr.Textbox(label="Telegram channel (optional)",
                                        placeholder="channel_name" if tg_ok else "add .env creds to enable",
                                        interactive=tg_ok)
                with gr.Row():
                    tg_count = gr.Slider(1, 30, value=5, step=1, label="max", scale=2)
                    tg_btn = gr.Button("Scrape", interactive=tg_ok, scale=1, size="sm")
        ingest_status = gr.Markdown()

        pool = gr.CheckboxGroup(choices=_pool_choices(), value=_default_pool(),
                                label="Source pool — order = strip rotation order (N videos)")

        with gr.Row():
            sw = gr.Slider(1, 64, value=2, step=1, label="Strip width (px)")
            step = gr.Slider(-10, 10, value=1, step=1, label="Frame step (per strip)")
            orient = gr.Radio(["vertical", "horizontal"], value="vertical",
                              label="Orientation")
            mode = gr.Radio(["rotate", "single"], value="rotate",
                            label="Sources", info="rotate: strip i ← video i mod N · single: slit-scan one clip")

        with gr.Row():
            fb_on = gr.Checkbox(value=False, label="Feedback trails", scale=1)
            fb_decay = gr.Slider(0.0, 0.98, value=0.8, step=0.02, label="Persistence", scale=2)
            blur_type = gr.Dropdown(BLUR_CHOICES, value="(none)", label="Blur (post)", scale=2)
            blur_amt = gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Blur amount", scale=2)

        with gr.Row():
            width = gr.Slider(160, 1280, value=config.DEFAULT_WIDTH, step=16, label="Width")
            height = gr.Slider(90, 720, value=config.DEFAULT_HEIGHT, step=2, label="Height")
            fps = gr.Slider(4, 30, value=config.DEFAULT_FPS, step=1, label="FPS")

        with gr.Row():
            preview_btn = gr.Button("Preview", variant="secondary")
            render_btn = gr.Button("Render", variant="primary")
        status = gr.Markdown()
        with gr.Row():
            video = gr.Video(label="Output", autoplay=True)
            download = gr.File(label="Download")

        inputs = [pool, sw, step, orient, mode, fb_on, fb_decay,
                  blur_type, blur_amt, width, height, fps]
        preview_btn.click(do_preview, inputs, [video, status])
        render_btn.click(do_render, inputs, [video, download, status])
        up.upload(do_upload, [up, pool], [pool, ingest_status])
        tg_btn.click(do_scrape, [tg_channel, tg_count, pool], [pool, ingest_status])

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1",
                allowed_paths=[config.OUTPUT_DIR, config.LIBRARY_DIR])
