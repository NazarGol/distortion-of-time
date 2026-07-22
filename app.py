"""
Distortion of Time — Gradio UI.

Tabs:
  1. Import / Scrape  — credential-free local import (+ optional Telegram scrape)
  2. Library          — thumbnails of clips currently available
  3. Compose          — layer clips with temporal offsets, then an ordered effect
                        chain (Combine → Feedback → Blur), Preview / Render,
                        save / load presets.

Run:  python app.py    (inside the project venv)
The compositor never needs Telegram credentials — a colleague can run everything
here on a plain folder of clips.
"""
from __future__ import annotations

import glob
import json
import os
import time

import gradio as gr

import config
from composition import Composition, new_render_path
from core import device_label
from frames import clip as clipmod
from ingest import local as ingest_local
from ingest import telegram as ingest_tg
from render.writer import write_mp4

MAX_LAYERS = 4
COMBINE_EFFECTS = ["interlace", "superimpose"]
BLUR_EFFECTS = ["temporal_average", "motion_flow_blur", "gaussian_blur",
                "fft_lowpass", "luminance_blur"]


# ── shared helpers ───────────────────────────────────────────────────────────────
def _clip_names() -> list[str]:
    return ingest_local.library_names()


def _gallery_items():
    items = []
    for path in ingest_local.list_library():
        try:
            thumb = clipmod.thumbnail(path)
        except Exception:
            thumb = path
        items.append((thumb, os.path.basename(path)))
    return items


def _preset_names() -> list[str]:
    return sorted(os.path.basename(p) for p in glob.glob(os.path.join(config.COMPOSITIONS_DIR, "*.json")))


def _default_clips():
    names = _clip_names()
    first = names[0] if names else None
    second = names[1] if len(names) > 1 else first
    return first, second


# ── Build a Composition from the flat UI inputs ──────────────────────────────────
# Input order (kept in lock-step with `all_inputs` at wiring time):
#   [0:3]  width, height, fps
#   [3:10] combine_mode, strip_w, orient, phase, feather, sup_blend, sup_mix
#   [10:13] fb_enabled, fb_decay, fb_mode
#   [13:22] blur_type, ta_window, mf_taps, mf_strength, gb_sigma, fft_cutoff,
#           lum_maxsigma, lum_levels, lum_invert
#   [22:22+5*MAX_LAYERS] per layer: enabled, clip, offset, scale, opacity
def _compose(args) -> Composition:
    width, height, fps = args[0:3]
    (combine_mode, strip_w, orient, phase, feather, sup_blend, sup_mix) = args[3:10]
    (fb_enabled, fb_decay, fb_mode) = args[10:13]
    (blur_type, ta_window, mf_taps, mf_strength, gb_sigma, fft_cutoff,
     lum_maxsigma, lum_levels, lum_invert) = args[13:22]
    layer_args = args[22:22 + 5 * MAX_LAYERS]

    layers = []
    for k in range(MAX_LAYERS):
        en, clip, off, scale, op = layer_args[k * 5: k * 5 + 5]
        if en and clip:
            layers.append({"clip": clip, "offset": int(off),
                           "scale": float(scale), "opacity": float(op)})

    chain = []
    if len(layers) >= 2 and combine_mode == "interlace":
        chain.append({"name": "interlace", "params": {
            "strip_width": int(strip_w), "orientation": orient,
            "phase_speed": float(phase), "feather": int(feather)}})
    elif len(layers) >= 2 and combine_mode == "superimpose":
        chain.append({"name": "superimpose", "params": {
            "blend": sup_blend, "mix": float(sup_mix)}})
    # combine_mode == "(none)" -> layers mean-flattened by the renderer

    if fb_enabled:
        chain.append({"name": "feedback", "params": {
            "decay": float(fb_decay), "mode": fb_mode}})

    blur_params = {
        "temporal_average": {"window": int(ta_window)},
        "motion_flow_blur": {"taps": int(mf_taps), "strength": float(mf_strength)},
        "gaussian_blur": {"sigma": float(gb_sigma)},
        "fft_lowpass": {"cutoff": float(fft_cutoff)},
        "luminance_blur": {"max_sigma": float(lum_maxsigma),
                           "levels": int(lum_levels), "invert": bool(lum_invert)},
    }
    if blur_type in blur_params:
        chain.append({"name": blur_type, "params": blur_params[blur_type]})

    return Composition(layers=layers, chain=chain,
                       width=int(width), height=int(height), fps=int(fps))


def do_preview(*args):
    comp = _compose(args)
    if not comp.layers:
        raise gr.Error("Enable at least one layer with a clip selected.")
    t0 = time.time()
    frames, ctx = comp.render(is_preview=True)
    out = os.path.join(config.OUTPUT_DIR, f"preview_{int(time.time())}.mp4")
    write_mp4(frames, out, fps=comp.fps)
    chain_desc = " → ".join(e["name"] for e in comp.chain) or "passthrough"
    return out, (f"Preview · {len(frames)}f @ {ctx.width}×{ctx.height} · "
                 f"{len(comp.layers)} layer(s) · [{chain_desc}] · "
                 f"{device_label()} · {time.time()-t0:.1f}s")


def do_render(*args):
    comp = _compose(args)
    if not comp.layers:
        raise gr.Error("Enable at least one layer with a clip selected.")
    t0 = time.time()
    frames, ctx = comp.render(is_preview=False)
    out = new_render_path("render")
    write_mp4(frames, out, fps=comp.fps)
    chain_desc = " → ".join(e["name"] for e in comp.chain) or "passthrough"
    return out, out, (f"Rendered · {len(frames)}f @ {ctx.width}×{ctx.height} · "
                      f"{len(comp.layers)} layer(s) · [{chain_desc}] · "
                      f"{device_label()} · {time.time()-t0:.1f}s\n{out}")


# ── save / load presets ──────────────────────────────────────────────────────────
def do_save(name, *args):
    comp = _compose(args)
    comp.name = (name or "composition").strip() or "composition"
    if not comp.layers:
        raise gr.Error("Nothing to save — enable at least one layer first.")
    path = comp.save()
    return (path, f"Saved preset → {os.path.basename(path)}",
            gr.update(choices=_preset_names(), value=os.path.basename(path)))


def do_load_render(preset_name):
    if not preset_name:
        raise gr.Error("Pick a saved preset first.")
    path = os.path.join(config.COMPOSITIONS_DIR, preset_name)
    comp = Composition.load(path)
    if not comp.layers:
        raise gr.Error("That preset has no layers.")
    t0 = time.time()
    frames, ctx = comp.render(is_preview=False)
    out = new_render_path("render")
    write_mp4(frames, out, fps=comp.fps)
    chain_desc = " → ".join(e["name"] for e in comp.chain) or "passthrough"
    return (out, out, comp.to_dict(),
            f"Loaded & rendered '{comp.name}' · [{chain_desc}] · {time.time()-t0:.1f}s")


# ── Import / Scrape callbacks ────────────────────────────────────────────────────
def do_import_files(files):
    paths = [f if isinstance(f, str) else getattr(f, "name", None) for f in (files or [])]
    imported = ingest_local.import_files([p for p in paths if p])
    return _after_library_change(f"Imported {len(imported)} file(s): {', '.join(imported) or '—'}")


def do_import_folder(folder):
    imported = ingest_local.import_folder(folder)
    return _after_library_change(f"Imported {len(imported)} from folder: {', '.join(imported) or '—'}")


def do_scrape(channel, count, date_from, date_to, max_mb):
    msg = ingest_tg.scrape(channel, int(count), date_from, date_to, float(max_mb))
    return _after_library_change(msg)


def _after_library_change(status_msg):
    names = _clip_names()
    updates = [gr.update(choices=names) for _ in range(MAX_LAYERS)]
    return (status_msg, _gallery_items(), *updates)


def refresh_library():
    return _gallery_items()


# ── UI ───────────────────────────────────────────────────────────────────────────
def build_ui():
    names = _clip_names()
    d1, d2 = _default_clips()
    tg_ok, tg_msg = ingest_tg.availability()

    with gr.Blocks(title="Distortion of Time") as demo:
        gr.Markdown(
            "# Distortion of Time\n"
            "Weave degraded footage into strips that hold *several moments at once*. "
            f"Compute: **{device_label()}**."
        )

        # ── Tab 1: Import / Scrape ───────────────────────────────────────────────
        with gr.Tab("Import / Scrape"):
            gr.Markdown("### Local import  ·  no credentials needed")
            up = gr.File(label="Upload clips", file_count="multiple", file_types=["video"])
            import_btn = gr.Button("Import uploaded files", variant="primary")
            with gr.Row():
                folder_in = gr.Textbox(label="…or import every video in a folder",
                                       placeholder="/path/to/clips", scale=3)
                folder_btn = gr.Button("Import folder", scale=1)
            import_status = gr.Markdown()

            gr.Markdown("---\n### Telegram scrape  ·  optional")
            tg_status = gr.Markdown(tg_msg)
            with gr.Row():
                tg_channel = gr.Textbox(label="Channel (public username)",
                                        placeholder="some_channel", scale=2)
                tg_count = gr.Slider(1, 50, value=5, step=1, label="Max videos")
                tg_mb = gr.Slider(5, 200, value=50, step=5, label="Max size (MB)")
            with gr.Row():
                tg_from = gr.Textbox(value="2022-01-01", label="From (YYYY-MM-DD)")
                tg_to = gr.Textbox(value="2030-01-01", label="To (YYYY-MM-DD)")
            tg_btn = gr.Button("Scrape into library", interactive=tg_ok)

        # ── Tab 2: Library ───────────────────────────────────────────────────────
        with gr.Tab("Library"):
            refresh_btn = gr.Button("Refresh")
            gallery = gr.Gallery(value=_gallery_items(), label="Clips in library/",
                                 columns=4, height=380, allow_preview=False)

        # ── Tab 3: Compose ───────────────────────────────────────────────────────
        with gr.Tab("Compose"):
            gr.Markdown(
                "### Layers\n"
                "Enable 2+ layers. **Temporal offset** shifts a layer to a different "
                "moment (wraps around the clip). *Use the same clip in two layers with "
                "different offsets* to turn one clip into many moments."
            )
            layer_en, layer_clip, layer_off, layer_scale, layer_op = [], [], [], [], []
            for i in range(MAX_LAYERS):
                with gr.Row():
                    default_clip = d1 if i == 0 else (d2 if i == 1 else None)
                    en = gr.Checkbox(value=(i < 2), label=f"L{i+1}", scale=0, min_width=60)
                    cd = gr.Dropdown(choices=names, value=default_clip, label="Clip", scale=3)
                    of = gr.Slider(0, 120, value=(0 if i == 0 else 6 * i), step=1,
                                   label="Offset (frames)", scale=2)
                    sc = gr.Slider(0.5, 2.0, value=1.0, step=0.05, label="Scale", scale=1)
                    op = gr.Slider(0.0, 1.0, value=1.0, step=0.05, label="Opacity", scale=1)
                layer_en.append(en); layer_clip.append(cd); layer_off.append(of)
                layer_scale.append(sc); layer_op.append(op)

            # ── Combine stage ────────────────────────────────────────────────────
            gr.Markdown("### 1 · Combine layers")
            combine_mode = gr.Dropdown(choices=["interlace", "superimpose", "(none)"],
                                       value="interlace", label="Combine mode")
            with gr.Group(visible=True) as interlace_grp:
                with gr.Row():
                    strip_w = gr.Slider(1, 64, value=2, step=1, label="Strip width (px)")
                    orient = gr.Dropdown(choices=["vertical", "horizontal"],
                                         value="vertical", label="Orientation")
                    phase = gr.Slider(-8, 8, value=1.0, step=0.25, label="Scroll (px/frame)")
                    feather = gr.Slider(0, 16, value=1, step=1, label="Feather (px)")
            with gr.Group(visible=False) as superimpose_grp:
                with gr.Row():
                    sup_blend = gr.Dropdown(
                        choices=["normal", "add", "screen", "multiply", "difference"],
                        value="screen", label="Blend mode")
                    sup_mix = gr.Slider(0, 1, value=0.6, step=0.05, label="Mix (upper layers)")

            def _toggle_combine(mode):
                return (gr.update(visible=(mode == "interlace")),
                        gr.update(visible=(mode == "superimpose")))
            combine_mode.change(_toggle_combine, [combine_mode],
                                [interlace_grp, superimpose_grp])

            # ── Feedback stage ───────────────────────────────────────────────────
            gr.Markdown("### 2 · Feedback trails  (hyper-imposition)")
            with gr.Row():
                fb_enabled = gr.Checkbox(value=False, label="Enable trails", scale=0, min_width=120)
                fb_decay = gr.Slider(0.0, 0.98, value=0.8, step=0.02, label="Persistence")
                fb_mode = gr.Dropdown(choices=["max", "mean", "add"], value="max", label="Trail mode")

            # ── Blur stage ───────────────────────────────────────────────────────
            gr.Markdown("### 3 · Blur  (temporal smear / soften)")
            blur_type = gr.Dropdown(choices=["(none)"] + BLUR_EFFECTS, value="(none)",
                                    label="Blur type")
            with gr.Group(visible=False) as ta_grp:
                ta_window = gr.Slider(1, 24, value=4, step=1, label="Temporal window (frames)")
            with gr.Group(visible=False) as mf_grp:
                with gr.Row():
                    mf_taps = gr.Slider(1, 6, value=3, step=1, label="Flow taps/side")
                    mf_strength = gr.Slider(0.2, 4.0, value=1.0, step=0.1, label="Flow strength")
            with gr.Group(visible=False) as gb_grp:
                gb_sigma = gr.Slider(0.0, 20.0, value=3.0, step=0.5, label="Gaussian sigma (px)")
            with gr.Group(visible=False) as fft_grp:
                fft_cutoff = gr.Slider(0.02, 1.0, value=0.25, step=0.02, label="FFT keep fraction")
            with gr.Group(visible=False) as lum_grp:
                with gr.Row():
                    lum_maxsigma = gr.Slider(1.0, 24.0, value=8.0, step=1.0, label="Max sigma (px)")
                    lum_levels = gr.Slider(2, 8, value=5, step=1, label="Blur levels")
                    lum_invert = gr.Checkbox(value=False, label="Blur darks instead")

            _blur_groups = {"temporal_average": ta_grp, "motion_flow_blur": mf_grp,
                            "gaussian_blur": gb_grp, "fft_lowpass": fft_grp,
                            "luminance_blur": lum_grp}

            def _toggle_blur(sel):
                return [gr.update(visible=(name == sel)) for name in BLUR_EFFECTS]
            blur_type.change(_toggle_blur, [blur_type],
                             [_blur_groups[n] for n in BLUR_EFFECTS])

            # ── Working space + actions ──────────────────────────────────────────
            gr.Markdown("### Working space")
            with gr.Row():
                width = gr.Slider(160, 1280, value=config.DEFAULT_WIDTH, step=16, label="Width")
                height = gr.Slider(90, 720, value=config.DEFAULT_HEIGHT, step=8, label="Height")
                fps = gr.Slider(4, 30, value=config.DEFAULT_FPS, step=1, label="FPS")

            with gr.Row():
                preview_btn = gr.Button("Preview (short, low-res)", variant="secondary")
                render_btn = gr.Button("Render (full)", variant="primary")
            status = gr.Markdown()
            with gr.Row():
                preview_vid = gr.Video(label="Preview", autoplay=True)
                render_vid = gr.Video(label="Render")
            download = gr.File(label="Download render")

            gr.Markdown("### Presets  (save / load composition)")
            with gr.Row():
                preset_name = gr.Textbox(label="Preset name", placeholder="my_look", scale=2)
                save_btn = gr.Button("Save preset", scale=1)
            with gr.Row():
                preset_dd = gr.Dropdown(choices=_preset_names(), label="Saved presets", scale=2)
                load_btn = gr.Button("Load & render", scale=1)
            preset_file = gr.File(label="Saved preset file")
            preset_json = gr.JSON(label="Loaded composition")

        # ── flat input list (order MUST match `_compose`) ────────────────────────
        common = [width, height, fps,
                  combine_mode, strip_w, orient, phase, feather, sup_blend, sup_mix,
                  fb_enabled, fb_decay, fb_mode,
                  blur_type, ta_window, mf_taps, mf_strength, gb_sigma, fft_cutoff,
                  lum_maxsigma, lum_levels, lum_invert]
        layer_inputs = []
        for i in range(MAX_LAYERS):
            layer_inputs += [layer_en[i], layer_clip[i], layer_off[i],
                             layer_scale[i], layer_op[i]]
        all_inputs = common + layer_inputs

        # ── wiring ────────────────────────────────────────────────────────────────
        import_btn.click(do_import_files, [up],
                         [import_status, gallery, *layer_clip])
        folder_btn.click(do_import_folder, [folder_in],
                         [import_status, gallery, *layer_clip])
        tg_btn.click(do_scrape, [tg_channel, tg_count, tg_from, tg_to, tg_mb],
                     [tg_status, gallery, *layer_clip])
        refresh_btn.click(refresh_library, outputs=[gallery])

        preview_btn.click(do_preview, all_inputs, [preview_vid, status])
        render_btn.click(do_render, all_inputs, [render_vid, download, status])
        save_btn.click(do_save, [preset_name, *all_inputs],
                       [preset_file, status, preset_dd])
        load_btn.click(do_load_render, [preset_dd],
                       [render_vid, download, preset_json, status])

    return demo


if __name__ == "__main__":
    print(f"[Distortion of Time] compute: {device_label()}")
    demo = build_ui()
    demo.launch(
        server_name="127.0.0.1",
        theme=gr.themes.Base(),
        allowed_paths=[config.OUTPUT_DIR, config.LIBRARY_DIR, config.CACHE_DIR,
                       config.COMPOSITIONS_DIR],
        show_error=True,
    )
