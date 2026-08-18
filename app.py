import os
import random
import time
import shutil
import re
import uuid

import torch
import numpy as np
from PIL import Image

from nodes import NODE_CLASS_MAPPINGS


# ============================================================
# LTX-2.5 VIDEO GENERATOR + MULTIPLE LORA
# ============================================================

print("\n" + "=" * 60)
print("        LTX-2.5 Video Generator + Multiple LoRA")
print("=" * 60)


# ============================================================
# COMFYUI NODES
# ============================================================

# Base Loaders
UNETLoader = NODE_CLASS_MAPPINGS["UNETLoader"]()
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
VAELoader = NODE_CLASS_MAPPINGS["VAELoader"]()

# Use 'LatentUpscaleModelLoader' which corresponds to the "Load Latent Upscale Model" node in the workflow
LatentUpscaleModelLoader = NODE_CLASS_MAPPINGS["LatentUpscaleModelLoader"]()

LoraLoader = NODE_CLASS_MAPPINGS["LoraLoader"]()

# Conditioning & Latent
CLIPTextEncode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
LTXVConditioning = NODE_CLASS_MAPPINGS["LTXVConditioning"]()
EmptyLTXVLatentVideo = NODE_CLASS_MAPPINGS["EmptyLTXVLatentVideo"]()
LTXVEmptyLatentAudio = NODE_CLASS_MAPPINGS["LTXVEmptyLatentAudio"]()
LTXVConcatAVLatent = NODE_CLASS_MAPPINGS["LTXVConcatAVLatent"]()
LTXVSeparateAVLatent = NODE_CLASS_MAPPINGS["LTXVSeparateAVLatent"]()
LTXVLatentUpsampler = NODE_CLASS_MAPPINGS["LTXVLatentUpsampler"]()

# Sampling
LTXVDualCFGGuider = NODE_CLASS_MAPPINGS["LTXVDualCFGGuider"]()
RandomNoise = NODE_CLASS_MAPPINGS["RandomNoise"]()
KSamplerSelect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
ManualSigmas = NODE_CLASS_MAPPINGS["ManualSigmas"]()
SamplerCustomAdvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()

# Decoding & Video Output
VAEDecodeTiled = NODE_CLASS_MAPPINGS["VAEDecodeTiled"]()
LTXVAudioVAEDecode = NODE_CLASS_MAPPINGS["LTXVAudioVAEDecode"]()
CreateVideo = NODE_CLASS_MAPPINGS["CreateVideo"]()
SaveVideo = NODE_CLASS_MAPPINGS["SaveVideo"]()


# ============================================================
# BASE MODELS
# ============================================================

startup_start = time.time()

with torch.inference_mode():

    print("\n[1/5] Loading UNet (LTX-2.5 Distilled)... ", end="", flush=True)
    t0 = time.time()
    base_model = UNETLoader.load_unet(
        "ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "default"
    )[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[2/5] Loading CLIP (Gemma 4 12B)... ", end="", flush=True)
    t0 = time.time()
    base_clip = CLIPLoader.load_clip(
        "gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        type="ltxv",
        device="default"
    )[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[3/5] Loading Video VAE... ", end="", flush=True)
    t0 = time.time()
    video_vae = VAELoader.load_vae("ltx-2.5-video-vae-bf16.safetensors")[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[4/5] Loading Audio VAE... ", end="", flush=True)
    t0 = time.time()
    audio_vae = VAELoader.load_vae("ltx-2.5-audio-vae-bf16.safetensors")[0]
    print(f"done ({time.time() - t0:.1f}s)")

    print("[5/5] Loading Latent Upscaler... ", end="", flush=True)
    t0 = time.time()
    upscale_model = LatentUpscaleModelLoader.load_model(
        "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
    )[0]
    print(f"done ({time.time() - t0:.1f}s)")

print(f"\n✅ Base models loaded in {time.time() - startup_start:.1f}s")
print("=" * 60)


# ============================================================
# LORA DIRECTORY
# ============================================================

LORA_DIR = "./models/loras"

def get_lora_files():
    if not os.path.exists(LORA_DIR):
        print(f"\n⚠️ LoRA directory not found:\n{os.path.abspath(LORA_DIR)}")
        return [""]
    
    files = []
    for root, dirs, filenames in os.walk(LORA_DIR):
        for filename in filenames:
            if filename.lower().endswith((".safetensors", ".pt", ".ckpt")):
                relative_path = os.path.relpath(os.path.join(root, filename), LORA_DIR)
                files.append(relative_path)
                
    files.sort()
    if not files:
        print(f"\n⚠️ No LoRAs found in:\n{os.path.abspath(LORA_DIR)}")
        return [""]
        
    print(f"\n✅ Found {len(files)} LoRA(s)")
    for f in files:
        print(f"   • {f}")
    return files

LORA_FILES = get_lora_files()


# ============================================================
# APPLY MULTIPLE LORAS
# ============================================================

def apply_loras(lora_names, lora_strengths, clip_strengths):
    model = base_model
    clip = base_clip
    applied = []
    
    for i in range(len(lora_names)):
        lora_name = lora_names[i]
        if not lora_name:
            continue
            
        model_strength = float(lora_strengths[i])
        clip_strength = float(clip_strengths[i])
        
        if model_strength == 0 and clip_strength == 0:
            continue
            
        print(f"\n   [{i + 1}] Applying LoRA: {lora_name} (M: {model_strength}, C: {clip_strength})")
        t0 = time.time()
        model, clip = LoraLoader.load_lora(model, clip, lora_name, model_strength, clip_strength)
        print(f"       done ({time.time() - t0:.1f}s)")
        applied.append(lora_name)
        
    return model, clip, applied


# ============================================================
# SAVE HELPERS
# ============================================================

save_dir = "./output"
os.makedirs(save_dir, exist_ok=True)

def get_save_path(prompt):
    safe_prompt = re.sub(r"[^a-zA-Z0-9_-]", "_", prompt)[:25]
    uid = uuid.uuid4().hex[:6]
    filename = f"{safe_prompt}_{uid}.mp4"
    return os.path.join(save_dir, filename)


# ============================================================
# GENERATION
# ============================================================

@torch.inference_mode()
def generate(input):
    values = input["input"]
    
    # Basic Settings
    positive_prompt = values["positive_prompt"]
    negative_prompt = values["negative_prompt"]
    seed = values["seed"]
    width = values["width"]
    height = values["height"]
    duration = values["duration"]
    frame_rate = values["frame_rate"]
    video_cfg = values["video_cfg"]
    audio_cfg = values["audio_cfg"]
    
    # LoRA Settings
    lora_names = values["lora_names"]
    lora_strengths = values["lora_strengths"]
    clip_strengths = values["clip_strengths"]
    
    print("\n" + "=" * 60)
    print("              NEW VIDEO GENERATION")
    print("=" * 60)
    
    total_start = time.time()
    
    # Calculate frame length
    length = int(duration) * int(frame_rate) + 1
    
    # 1. APPLY LORAS
    print("\n[1/7] Applying LoRAs...")
    t0 = time.time()
    model, clip, applied_loras = apply_loras(lora_names, lora_strengths, clip_strengths)
    print(f"✅ Applied {len(applied_loras)} LoRA(s) | Time: {time.time() - t0:.1f}s")
    
    # 2. PROMPTS & CONDITIONING
    print("\n[2/7] Encoding prompts & conditioning... ", end="", flush=True)
    t0 = time.time()
    positive = CLIPTextEncode.encode(clip, positive_prompt)[0]
    negative = CLIPTextEncode.encode(clip, negative_prompt)[0]
    positive_cond, negative_cond = LTXVConditioning.get_conditioning(
        positive, negative, float(frame_rate)
    )
    print(f"done ({time.time() - t0:.1f}s)")
    
    # 3. EMPTY LATENTS (VIDEO + AUDIO)
    print(f"\n[3/7] Creating empty latents ({length} frames)... ", end="", flush=True)
    t0 = time.time()
    video_latent = EmptyLTXVLatentVideo.generate(int(width), int(height), length, 1)[0]
    audio_latent = LTXVEmptyLatentAudio.generate(audio_vae, length, int(frame_rate), 1)[0]
    av_latent = LTXVConcatAVLatent.concat(video_latent, audio_latent)[0]
    print(f"done ({time.time() - t0:.1f}s)")
    
    # 4. LOW RESOLUTION SAMPLING
    print(f"\n[4/7] Sampling Low-Resolution Video...")
    t0 = time.time()
    guider = LTXVDualCFGGuider.get_guider(model, positive_cond, negative_cond, float(video_cfg), float(audio_cfg))[0]
    noise = RandomNoise.get_noise(int(seed), "fixed")[0]
    sampler = KSamplerSelect.get_sampler("euler_ancestral")[0]
    sigmas_1 = ManualSigmas.get_sigmas("1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0")[0]
    latent_1 = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas_1, av_latent)[0]
    print(f"      Sampling done ({time.time() - t0:.1f}s)")
    
    # 5. UPSCALE LATENT
    print(f"\n[5/7] Upscaling Latent... ", end="", flush=True)
    t0 = time.time()
    v_latent_1, a_latent_1 = LTXVSeparateAVLatent.separate(latent_1)
    up_v_latent = LTXVLatentUpsampler.upscale(v_latent_1, upscale_model, video_vae)[0]
    up_av_latent = LTXVConcatAVLatent.concat(up_v_latent, a_latent_1)[0]
    print(f"done ({time.time() - t0:.1f}s)")
    
    # 6. HIGH RESOLUTION SAMPLING
    print(f"\n[6/7] Sampling High-Resolution Video...")
    t0 = time.time()
    sigmas_2 = ManualSigmas.get_sigmas("0.85, 0.7250, 0.4219, 0.0")[0]
    latent_2 = SamplerCustomAdvanced.sample(noise, guider, sampler, sigmas_2, up_av_latent)[0]
    print(f"      Sampling done ({time.time() - t0:.1f}s)")
    
    # 7. DECODE & SAVE VIDEO
    print(f"\n[7/7] Decoding Video & Audio... ", end="", flush=True)
    t0 = time.time()
    v_latent_2, a_latent_2 = LTXVSeparateAVLatent.separate(latent_2)
    images = VAEDecodeTiled.decode(v_latent_2, video_vae, 512, 64, 64, 16)[0]
    audio_out = LTXVAudioVAEDecode.decode(a_latent_2, audio_vae)[0]
    video_obj = CreateVideo.create_video(images, audio_out, int(frame_rate), 8)[0]
    print(f"done ({time.time() - t0:.1f}s)")
    
    # Save Video File
    saved_video_info = SaveVideo.save_video(video_obj, "LTX_2.5_t2v", "auto", "auto")[0]
    video_path = os.path.join(save_dir, saved_video_info.get("subfolder", ""), saved_video_info["filename"])
    
    print(f"\n💾 Saved:\n   {video_path}")
    
    # Copy to Google Drive if mounted
    drive_path = "/content/gdrive/MyDrive/ltx_2_5"
    if os.path.exists("/content/gdrive/MyDrive"):
        os.makedirs(drive_path, exist_ok=True)
        shutil.copy(video_path, drive_path)
        print(f"☁️ Copied to Google Drive:\n   {drive_path}")
        
    # Summary
    print(f"\n🎨 LoRAs used:")
    if applied_loras:
        for lora in applied_loras: print(f"   • {lora}")
    else:
        print("   • None")
        
    print(f"\n🌱 Seed: {seed}")
    print(f"⏱️ Total: {time.time() - total_start:.1f}s")
    print("=" * 60 + "\n")
    
    return video_path, seed


# ============================================================
# GRADIO UI
# ============================================================

import gradio as gr

def generate_ui(
    positive_prompt, negative_prompt,
    width, height, duration, frame_rate,
    seed, video_cfg, audio_cfg,
    lora1, lora1_strength, lora1_clip,
    lora2, lora2_strength, lora2_clip,
    lora3, lora3_strength, lora3_clip,
    lora4, lora4_strength, lora4_clip,
    lora5, lora5_strength, lora5_clip
):
    lora_names = [lora1, lora2, lora3, lora4, lora5]
    lora_strengths = [lora1_strength, lora2_strength, lora3_strength, lora4_strength, lora5_strength]
    clip_strengths = [lora1_clip, lora2_clip, lora3_clip, lora4_clip, lora5_clip]
    
    input_data = {
        "input": {
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "width": int(width),
            "height": int(height),
            "duration": int(duration),
            "frame_rate": int(frame_rate),
            "seed": int(seed),
            "video_cfg": float(video_cfg),
            "audio_cfg": float(audio_cfg),
            "lora_names": lora_names,
            "lora_strengths": lora_strengths,
            "clip_strengths": clip_strengths
        }
    }
    
    video_path, used_seed = generate(input_data)
    return video_path, video_path, used_seed


DEFAULT_POSITIVE = """
Dynamic cinematic close-up of high-tech modular machinery self-assembling in midair, 
precision robotic parts, magnetic connectors, and glowing circuits clicking together, 
subtle smoke and light flares, extremely detailed titanium textures. 
The final product displays a clean, clear surface with large glowing engraved text 
“LTX-2.5” centered and unobstructed, dramatic lighting, photorealism, 8K, sharp focus.
"""

DEFAULT_NEGATIVE = "pc game, console game, video game, cartoon, childish, ugly, blurry, low quality"

custom_css = """
.gradio-container { font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif; }
.lora-box { border: 1px solid #888; border-radius: 10px; padding: 10px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as demo:
    gr.HTML("""
<div style="width:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; margin:20px 0;">
    <h1 style="font-size:2.5em; margin-bottom:10px;">LTX-2.5 Video + Multiple LoRA</h1>
</div>
""")
    
    with gr.Row():
        with gr.Column():
            positive = gr.Textbox(DEFAULT_POSITIVE, label="Positive Prompt", lines=6)
            negative = gr.Textbox(DEFAULT_NEGATIVE, label="Negative Prompt", lines=3)
            
            with gr.Row():
                width = gr.Number(value=1280, label="Width", precision=0)
                height = gr.Number(value=720, label="Height", precision=0)
                duration = gr.Number(value=5, label="Duration (seconds)", precision=0)
                frame_rate = gr.Number(value=24, label="Frame Rate (FPS)", precision=0)
                
            with gr.Row():
                seed = gr.Number(value=0, label="Seed (0 = random)", precision=0)
                video_cfg = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Video CFG")
                audio_cfg = gr.Slider(0.1, 10.0, value=1.0, step=0.1, label="Audio CFG")
                
            with gr.Accordion("🎨 LoRA Settings", open=True):
                gr.Markdown("### Stack multiple LoRAs\nLeave a slot empty if you don't want to use it.")
                
                for i in range(1, 6):
                    with gr.Group(elem_classes="lora-box"):
                        gr.Markdown(f"### LoRA {i}")
                        lora = gr.Dropdown(choices=([""] + LORA_FILES) if i > 1 else LORA_FILES, 
                                           value="" if i > 1 else LORA_FILES[0], label="LoRA")
                        with gr.Row():
                            m_strength = gr.Slider(-9.0, 9.0, value=1, step=0.05, label="Model Strength")
                            c_strength = gr.Slider(-2.0, 2.0, value=1, step=0.05, label="CLIP Strength")
                            
                        # Dynamically create variables for inputs
                        if i == 1: lora1, lora1_strength, lora1_clip = lora, m_strength, c_strength
                        elif i == 2: lora2, lora2_strength, lora2_clip = lora, m_strength, c_strength
                        elif i == 3: lora3, lora3_strength, lora3_clip = lora, m_strength, c_strength
                        elif i == 4: lora4, lora4_strength, lora4_clip = lora, m_strength, c_strength
                        elif i == 5: lora5, lora5_strength, lora5_clip = lora, m_strength, c_strength
            
            run = gr.Button("🚀 Generate Video", variant="primary", size="lg")
            
        with gr.Column():
            output_vid = gr.Video(label="Generated Video", height=600)
            download_video = gr.File(label="Download Video")
            used_seed = gr.Textbox(label="Seed Used", interactive=False)
            
    run.click(
        fn=generate_ui,
        inputs=[
            positive, negative, width, height, duration, frame_rate, seed, video_cfg, audio_cfg,
            lora1, lora1_strength, lora1_clip,
            lora2, lora2_strength, lora2_clip,
            lora3, lora3_strength, lora3_clip,
            lora4, lora4_strength, lora4_clip,
            lora5, lora5_strength, lora5_clip
        ],
        outputs=[output_vid, download_video, used_seed]
    )

# ============================================================
# LAUNCH
# ============================================================

demo.launch(share=True, debug=True)
