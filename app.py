import os
import time
import shutil
import subprocess
import uuid
import re

import torch
import numpy as np

from nodes import NODE_CLASS_MAPPINGS


# ============================================================
# LTX-2.5 TEXT-TO-VIDEO (dual stage + synced audio)
# ============================================================

print("\n" + "=" * 60)
print("        LTX-2.5  —  Text to Video")
print("=" * 60)


# ============================================================
# GENERIC NODE RUNNER
#
# Instead of guessing method names on custom nodes (LTXV nodes
# live in a third-party extension, not comfy-core), we look up
# each node class's registered `FUNCTION` attribute and call it
# dynamically — exactly the way ComfyUI itself executes a graph.
# ============================================================

_node_instances = {}
import asyncio
import nodes

# nodes.py only exposes the small "core" node set on import.
# Everything in comfy_extras/ (LatentUpscaleModelLoader, LTXV* nodes,
# TextGenerateLTX2Prompt, etc.) is only registered by this call, which
# main.py normally runs for you at ComfyUI startup.
asyncio.run(nodes.init_extra_nodes())

def get_node(node_type):
    """Instantiate (once) and cache a NODE_CLASS_MAPPINGS class."""

    if node_type not in _node_instances:

        if node_type not in NODE_CLASS_MAPPINGS:
            raise RuntimeError(
                f"Node type '{node_type}' was not found in "
                f"NODE_CLASS_MAPPINGS. Make sure the LTXVideo "
                f"custom node pack is installed."
            )

        _node_instances[node_type] = NODE_CLASS_MAPPINGS[node_type]()

    return _node_instances[node_type]


def run_node(node_type, **kwargs):
    """Call a ComfyUI node the same way the graph executor does:
    look up its FUNCTION attribute and invoke it with kwargs."""

    instance = get_node(node_type)

    fn_name = NODE_CLASS_MAPPINGS[node_type].FUNCTION

    fn = getattr(instance, fn_name)

    return fn(**kwargs)


# ============================================================
# BASE MODELS (loaded once at startup)
# ============================================================

startup_start = time.time()

with torch.inference_mode():

    print("\n[1/6] Loading UNet (LTX-2.5 distilled transformer)... ", end="", flush=True)
    t0 = time.time()
    unet_name = "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors"
    model = run_node("UNETLoader", unet_name=unet_name, weight_dtype="default")[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[2/6] Loading text encoder (Gemma-4 12B, LTX-2.5)... ", end="", flush=True)
    t0 = time.time()
    clip_name = "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors"
    clip = run_node("CLIPLoader", clip_name=clip_name, type="ltxv", device="default")[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[3/6] Loading prompt-enhancer text encoder (Gemma-4 e2b)... ", end="", flush=True)
    t0 = time.time()
    clip_enhance_name = "gemma4_e2b_it_bf16.safetensors"
    try:
        clip_enhance = run_node(
            "CLIPLoader", clip_name=clip_enhance_name, type="ltxv", device="default"
        )[0]
        prompt_enhance_available = "TextGenerateLTX2Prompt" in NODE_CLASS_MAPPINGS
    except Exception as e:
        print(f"skipped ({e})")
        clip_enhance = None
        prompt_enhance_available = False
    else:
        print(f"done ({time.time() - t0:.1f}s)")

    print("[4/6] Loading video VAE... ", end="", flush=True)
    t0 = time.time()
    video_vae = run_node("VAELoader", vae_name="ltx-2.5-video-vae-bf16.safetensors")[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[5/6] Loading audio VAE... ", end="", flush=True)
    t0 = time.time()
    audio_vae = run_node("VAELoader", vae_name="ltx-2.5-audio-vae-bf16.safetensors")[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[6/6] Loading latent spatial upscaler (x2)... ", end="", flush=True)
    t0 = time.time()
    upscale_model = run_node(
        "LatentUpscaleModelLoader",
        model_name="ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    )[0]
    print(f"done ({time.time() - t0:.1f}s)")

print(f"\n✅ Base models loaded in {time.time() - startup_start:.1f}s")
print("=" * 60)


# ============================================================
# SAVE HELPERS
# ============================================================

save_dir = "./results"
os.makedirs(save_dir, exist_ok=True)


def get_save_path(prompt):

    safe_prompt = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt)[:25]
    uid = uuid.uuid4().hex[:6]
    return os.path.join(save_dir, f"{safe_prompt}_{uid}.mp4")


def round_to_multiple(value, multiple=32):
    return max(multiple, int(round(value / multiple)) * multiple)


def mux_audio_video(video_path, audio_path, output_path):
    """Combine a silent video and a wav file into one mp4 via ffmpeg.
    Falls back to the silent video if ffmpeg or the audio track is
    unavailable."""

    if not shutil.which("ffmpeg") or audio_path is None:
        shutil.copy(video_path, output_path)
        return False

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0 or not os.path.exists(output_path):
        shutil.copy(video_path, output_path)
        return False

    return True


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate(input):

    values = input["input"]

    positive_prompt = values["positive_prompt"]
    negative_prompt = values["negative_prompt"]
    prompt_enhance = values["prompt_enhance"]

    duration = int(values["duration"])
    width = round_to_multiple(int(values["width"]), 64)
    height = round_to_multiple(int(values["height"]), 64)
    frame_rate = int(values["frame_rate"])
    seed = int(values["seed"])

    video_cfg = float(values["video_cfg"])
    audio_cfg = float(values["audio_cfg"])

    sampler_name = values["sampler_name"]

    sigmas_stage1 = values["sigmas_stage1"]
    sigmas_stage2 = values["sigmas_stage2"]

    tile_size = int(values["tile_size"])
    tile_overlap = int(values["tile_overlap"])
    temporal_size = int(values["temporal_size"])
    temporal_overlap = int(values["temporal_overlap"])

    stage2_seed = int(values["stage2_seed"])

    print("\n" + "=" * 60)
    print("              NEW GENERATION")
    print("=" * 60)

    total_start = time.time()

    length = duration * frame_rate + 1

    low_width = round_to_multiple(width // 2, 32)
    low_height = round_to_multiple(height // 2, 32)

    # ========================================================
    # [1/9] PROMPT ENHANCEMENT (optional)
    # ========================================================

    print("\n[1/9] Prompt enhancement... ", end="", flush=True)
    t0 = time.time()

    effective_prompt = positive_prompt

    if prompt_enhance and prompt_enhance_available:

        try:
            generated = run_node(
                "TextGenerateLTX2Prompt",
                clip=clip_enhance,
                image=None,
                video=None,
                audio=None,
                prompt=positive_prompt,
                max_length=600,
                sampling_mode="on",
                **{
                    "sampling_mode.temperature": 0.7,
                    "sampling_mode.top_k": 64,
                    "sampling_mode.top_p": 0.95,
                    "sampling_mode.min_p": 0.05,
                    "sampling_mode.repetition_penalty": 1.15,
                    "sampling_mode.seed": 0,
                    "sampling_mode.presence_penalty": 0,
                },
                thinking=False,
                use_default_template=True,
            )[0]

            effective_prompt = generated

            print(f"done ({time.time() - t0:.1f}s)")

        except Exception as e:
            print(f"failed, using original prompt ({e})")

    else:
        print("skipped")

    # ========================================================
    # [2/9] ENCODE CONDITIONING
    # ========================================================

    print("[2/9] Encoding prompts... ", end="", flush=True)
    t0 = time.time()

    positive = run_node("CLIPTextEncode", clip=clip, text=effective_prompt)[0]
    negative = run_node("CLIPTextEncode", clip=clip, text=negative_prompt)[0]

    positive, negative = run_node(
        "LTXVConditioning", positive=positive, negative=negative, frame_rate=float(frame_rate)
    )

    print(f"done ({time.time() - t0:.1f}s)")

    # ========================================================
    # [3/9] LOW-RESOLUTION STAGE — sampling
    # ========================================================

    print(f"[3/9] Low-res pass ({low_width}x{low_height}, {length} frames)... ", end="", flush=True)
    t0 = time.time()

    video_latent_low = run_node(
        "EmptyLTXVLatentVideo", width=low_width, height=low_height, length=length, batch_size=1
    )[0]

    audio_latent = run_node(
        "LTXVEmptyLatentAudio",
        audio_vae=audio_vae,
        frames_number=length,
        frame_rate=frame_rate,
        batch_size=1,
    )[0]

    av_latent_low = run_node(
        "LTXVConcatAVLatent", video_latent=video_latent_low, audio_latent=audio_latent
    )[0]

    guider_low = run_node(
        "LTXVDualCFGGuider",
        model=model,
        positive=positive,
        negative=negative,
        video_cfg=video_cfg,
        audio_cfg=audio_cfg,
    )[0]

    sampler = run_node("KSamplerSelect", sampler_name=sampler_name)[0]

    sigmas_low = run_node("ManualSigmas", sigmas=sigmas_stage1)[0]

    noise_low = run_node("RandomNoise", noise_seed=seed)[0]

    sampled_low = run_node(
        "SamplerCustomAdvanced",
        noise=noise_low,
        guider=guider_low,
        sampler=sampler,
        sigmas=sigmas_low,
        latent_image=av_latent_low,
    )[0]

    video_latent_low_out, audio_latent_out = run_node(
        "LTXVSeparateAVLatent", av_latent=sampled_low
    )

    print(f"done ({time.time() - t0:.1f}s)")

    # ========================================================
    # [4/9] LATENT UPSCALE (x2)
    # ========================================================

    print(f"[4/9] Upscaling latent to {width}x{height}... ", end="", flush=True)
    t0 = time.time()

    video_latent_hi = run_node(
        "LTXVLatentUpsampler",
        samples=video_latent_low_out,
        upscale_model=upscale_model,
        vae=video_vae,
    )[0]

    av_latent_hi = run_node(
        "LTXVConcatAVLatent", video_latent=video_latent_hi, audio_latent=audio_latent_out
    )[0]

    print(f"done ({time.time() - t0:.1f}s)")

    # ========================================================
    # [5/9] HIGH-RESOLUTION STAGE — sampling
    # ========================================================

    print("[5/9] High-res refinement pass... ", end="", flush=True)
    t0 = time.time()

    guider_hi = run_node(
        "LTXVDualCFGGuider",
        model=model,
        positive=positive,
        negative=negative,
        video_cfg=video_cfg,
        audio_cfg=audio_cfg,
    )[0]

    sigmas_hi = run_node("ManualSigmas", sigmas=sigmas_stage2)[0]

    noise_hi = run_node("RandomNoise", noise_seed=stage2_seed)[0]

    sampled_hi = run_node(
        "SamplerCustomAdvanced",
        noise=noise_hi,
        guider=guider_hi,
        sampler=sampler,
        sigmas=sigmas_hi,
        latent_image=av_latent_hi,
    )[0]

    video_latent_final, audio_latent_final = run_node(
        "LTXVSeparateAVLatent", av_latent=sampled_hi
    )

    print(f"done ({time.time() - t0:.1f}s)")

    # ========================================================
    # [6/9] VAE DECODE (video, tiled)
    # ========================================================

    print("[6/9] Decoding video frames... ", end="", flush=True)
    t0 = time.time()

    images = run_node(
        "VAEDecodeTiled",
        samples=video_latent_final,
        vae=video_vae,
        tile_size=tile_size,
        overlap=tile_overlap,
        temporal_size=temporal_size,
        temporal_overlap=temporal_overlap,
    )[0].detach()

    print(f"done ({time.time() - t0:.1f}s)")

    # ========================================================
    # [7/9] VAE DECODE (audio)
    # ========================================================

    print("[7/9] Decoding audio... ", end="", flush=True)
    t0 = time.time()

    audio_path = None

    try:
        audio = run_node(
            "LTXVAudioVAEDecode", samples=audio_latent_final, audio_vae=audio_vae
        )[0]

        waveform = audio["waveform"].detach().cpu()
        sample_rate = int(audio["sample_rate"])

        wav_np = waveform[0].numpy()  # [channels, samples]
        wav_np = np.clip(wav_np, -1.0, 1.0)
        wav_int16 = (wav_np * 32767.0).astype(np.int16).T  # [samples, channels]

        audio_path = os.path.join(save_dir, f"_tmp_audio_{uuid.uuid4().hex[:6]}.wav")

        from scipy.io import wavfile
        wavfile.write(audio_path, sample_rate, wav_int16)

        print(f"done ({time.time() - t0:.1f}s)")

    except Exception as e:
        print(f"skipped, video will be silent ({e})")

    # ========================================================
    # [8/9] WRITE VIDEO + MUX AUDIO
    # ========================================================

    print("[8/9] Encoding video... ", end="", flush=True)
    t0 = time.time()

    frames_uint8 = (images.numpy() * 255).clip(0, 255).astype(np.uint8)

    silent_video_path = os.path.join(save_dir, f"_tmp_video_{uuid.uuid4().hex[:6]}.mp4")

    import imageio
    with imageio.get_writer(
        silent_video_path, fps=frame_rate, codec="libx264", quality=8
    ) as writer:
        for frame in frames_uint8:
            writer.append_data(frame)

    save_path = get_save_path(positive_prompt)
    has_audio = mux_audio_video(silent_video_path, audio_path, save_path)

    for tmp_file in (silent_video_path, audio_path):
        if tmp_file and os.path.exists(tmp_file):
            os.remove(tmp_file)

    print(f"done ({time.time() - t0:.1f}s)")

    # ========================================================
    # [9/9] SAVE
    # ========================================================

    print(f"\n💾 Saved:\n   {save_path}")
    print(f"🔊 Audio track: {'yes' if has_audio else 'no (silent)'}")

    drive_path = "/content/gdrive/MyDrive/ltx2_5_t2v"

    if os.path.exists(drive_path):
        shutil.copy(save_path, drive_path)
        print(f"☁️ Copied to Google Drive:\n   {drive_path}")

    print(f"\n📝 Prompt used: {effective_prompt[:120]}{'...' if len(effective_prompt) > 120 else ''}")
    print(f"🌱 Seed: {seed}")
    print(f"⏱️ Total: {time.time() - total_start:.1f}s")
    print("=" * 60 + "\n")

    return save_path, seed


# ============================================================
# GRADIO
# ============================================================

import gradio as gr


def generate_ui(
    positive_prompt,
    negative_prompt,
    prompt_enhance,
    duration,
    width,
    height,
    frame_rate,
    seed,
    video_cfg,
    audio_cfg,
    sampler_name,
    sigmas_stage1,
    sigmas_stage2,
    tile_size,
    tile_overlap,
    temporal_size,
    temporal_overlap,
    stage2_seed,
):

    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "prompt_enhance": prompt_enhance,
            "duration": duration,
            "width": width,
            "height": height,
            "frame_rate": frame_rate,
            "seed": seed,
            "video_cfg": video_cfg,
            "audio_cfg": audio_cfg,
            "sampler_name": sampler_name,
            "sigmas_stage1": sigmas_stage1,
            "sigmas_stage2": sigmas_stage2,
            "tile_size": tile_size,
            "tile_overlap": tile_overlap,
            "temporal_size": temporal_size,
            "temporal_overlap": temporal_overlap,
            "stage2_seed": stage2_seed,
        }
    }

    video_path, used_seed = generate(input_data)

    return video_path, video_path, used_seed


# ============================================================
# RESOLUTION PRESETS (16:9, multiple of 32 — see workflow note)
# ============================================================

RESOLUTION_PRESETS = {
    "0.2 MP (608x352) — fastest": (608, 352),
    "0.4 MP (864x480)": (864, 480),
    "0.6 MP (1056x608)": (1056, 608),
    "0.9 MP (1280x736) — default": (1280, 720),
    "1.2 MP (1504x832)": (1504, 832),
    "1.8 MP (1824x1024)": (1824, 1024),
    "2.0 MP (1920x1088) — slowest": (1920, 1088),
}


def apply_preset(preset_name):
    w, h = RESOLUTION_PRESETS[preset_name]
    return w, h


# ============================================================
# DEFAULT PROMPTS
# ============================================================

DEFAULT_POSITIVE = """A close-up of an Arctic hunter's face, eyes fixed straight ahead, frost dusting his dark beard, one hand slowly reaching toward the rifle slung on his back. The camera slowly pulls back, revealing a polar bear moving along a distant ice ridge, facing toward him. The wind carries the distant sound of shifting ice, a single low growl rolling across the water."""

DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, blurry, low quality, watermark, subtitles"

DEFAULT_SIGMAS_STAGE1 = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
DEFAULT_SIGMAS_STAGE2 = "0.85, 0.7250, 0.4219, 0.0"


# ============================================================
# CSS
# ============================================================

custom_css = """
.gradio-container {
    font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
}
"""


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:

    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
<h1 style="font-size:2.5em; margin-bottom:10px;">LTX-2.5 Text to Video</h1>
<p style="opacity:0.7;">Dual-stage pixel diffusion with synchronized audio</p>
</div>
""")

    with gr.Row():

        # ====================================================
        # LEFT
        # ====================================================

        with gr.Column():

            positive = gr.Textbox(DEFAULT_POSITIVE, label="Prompt", lines=6)
            negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)

            prompt_enhance = gr.Checkbox(
                value=True,
                label="✨ Prompt Enhance (expand short prompts into cinematic detail)",
            )

            with gr.Row():
                duration = gr.Slider(1, 10, value=5, step=1, label="Duration (seconds)")
                frame_rate = gr.Slider(8, 30, value=24, step=1, label="Frame Rate (fps)")

            with gr.Row():
                width = gr.Number(value=1280, label="Width", precision=0)
                height = gr.Number(value=720, label="Height", precision=0)

            resolution_preset = gr.Dropdown(
                choices=list(RESOLUTION_PRESETS.keys()),
                value="0.9 MP (1280x736) — default",
                label="Resolution Preset (16:9)",
            )

            seed = gr.Number(value=558811532553686, label="Seed", precision=0)

            with gr.Accordion("⚙️ Advanced Settings", open=False):

                with gr.Row():
                    video_cfg = gr.Slider(0.5, 8.0, value=1.0, step=0.1, label="Video CFG")
                    audio_cfg = gr.Slider(0.5, 8.0, value=1.0, step=0.1, label="Audio CFG")

                sampler_name = gr.Dropdown(
                    choices=["euler_ancestral", "euler", "dpmpp_2m", "dpmpp_2m_sde"],
                    value="euler_ancestral",
                    label="Sampler",
                )

                sigmas_stage1 = gr.Textbox(
                    DEFAULT_SIGMAS_STAGE1, label="Low-Res Sigmas (comma-separated)"
                )
                sigmas_stage2 = gr.Textbox(
                    DEFAULT_SIGMAS_STAGE2, label="High-Res Sigmas (comma-separated)"
                )

                stage2_seed = gr.Number(
                    value=42, label="High-Res Stage Seed (independent, per workflow default)", precision=0
                )

                gr.Markdown("**VAE Decode Tiling**")

                with gr.Row():
                    tile_size = gr.Number(value=512, label="Tile Size", precision=0)
                    tile_overlap = gr.Number(value=64, label="Tile Overlap", precision=0)

                with gr.Row():
                    temporal_size = gr.Number(value=64, label="Temporal Tile Size", precision=0)
                    temporal_overlap = gr.Number(value=16, label="Temporal Overlap", precision=0)

            run = gr.Button("🎬 Generate Video", variant="primary", size="lg")

        # ====================================================
        # RIGHT
        # ====================================================

        with gr.Column():

            output_video = gr.Video(label="Generated Video", height=500)
            download_video = gr.File(label="Download Video")
            used_seed = gr.Textbox(label="Seed Used", interactive=False)

    # ========================================================
    # EVENTS
    # ========================================================

    resolution_preset.change(
        fn=apply_preset, inputs=[resolution_preset], outputs=[width, height]
    )

    run.click(
        fn=generate_ui,
        inputs=[
            positive,
            negative,
            prompt_enhance,
            duration,
            width,
            height,
            frame_rate,
            seed,
            video_cfg,
            audio_cfg,
            sampler_name,
            sigmas_stage1,
            sigmas_stage2,
            tile_size,
            tile_overlap,
            temporal_size,
            temporal_overlap,
            stage2_seed,
        ],
        outputs=[output_video, download_video, used_seed],
    )


# ============================================================
# LAUNCH
# ============================================================

demo.launch(share=True, debug=True)
