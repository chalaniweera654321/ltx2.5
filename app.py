import os
import json
import uuid
import time
import random
import requests
import websocket
import gradio as gr


# ============================================================
# LTX-2.5 COMFYUI VIDEO APP
# No LoRAs
# ============================================================

COMFYUI_URL = "http://127.0.0.1:8188"

CLIENT_ID = str(uuid.uuid4())

# ------------------------------------------------------------
# Default settings from the supplied workflow
# ------------------------------------------------------------

DEFAULT_NEGATIVE = (
    "blurry, low quality, still frame, frames, watermark, "
    "overlay, titles, has blurbox, has subtitles"
)

DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720
DEFAULT_FPS = 24
DEFAULT_DURATION = 5

# The supplied workflow uses 105 frames at 25 FPS in one section,
# but the LTX-2.5 workflow description exposes duration/FPS.
# We calculate frames from duration and FPS.
def duration_to_frames(duration, fps):
    frames = int(round(float(duration) * int(fps)))

    # LTX video frame counts work best when represented as
    # 8n + 1.
    frames = max(9, frames)

    remainder = (frames - 1) % 8
    if remainder:
        frames += 8 - remainder

    return frames


# ============================================================
# WORKFLOW
# ============================================================

def build_workflow(
    prompt,
    negative_prompt,
    width,
    height,
    duration,
    fps,
    seed,
    prompt_enhance,
):
    frames = duration_to_frames(duration, fps)

    # --------------------------------------------------------
    # This is based on the supplied LTX workflow.
    #
    # IMPORTANT:
    # The exact node names/inputs can differ depending on the
    # installed ComfyUI/LTX version.
    # --------------------------------------------------------

    workflow = {
        "1": {
            "inputs": {
                "ckpt_name":
                    "ltx-av-step-1751000_vocoder_24K.safetensors"
            },
            "class_type": "CheckpointLoaderSimple"
        },

        "2": {
            "inputs": {
                "gemma_path":
                    "gemma-3-12b-it-qat-q4_0-unquantized_readout_proj/model/model.safetensors",
                "ltxv_path":
                    "ltx-av-step-1751000_vocoder_24K.safetensors",
                "max_length": 1024
            },
            "class_type": "LTXVGemmaCLIPModelLoader"
        },

        # ----------------------------------------------------
        # POSITIVE PROMPT
        # ----------------------------------------------------

        "3": {
            "inputs": {
                "text": prompt,
                "clip": [
                    "2",
                    0
                ]
            },
            "class_type": "CLIPTextEncode"
        },

        # ----------------------------------------------------
        # NEGATIVE PROMPT
        # ----------------------------------------------------

        "4": {
            "inputs": {
                "text": negative_prompt,
                "clip": [
                    "2",
                    0
                ]
            },
            "class_type": "CLIPTextEncode"
        },

        # ----------------------------------------------------
        # SAMPLER
        # ----------------------------------------------------

        "8": {
            "inputs": {
                "sampler_name": "euler"
            },
            "class_type": "KSamplerSelect"
        },

        # ----------------------------------------------------
        # SCHEDULER
        # ----------------------------------------------------

        "9": {
            "inputs": {
                "steps": 20,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stretch": True,
                "terminal": 0.1,
                "latent": [
                    "28",
                    0
                ]
            },
            "class_type": "LTXVScheduler"
        },

        # ----------------------------------------------------
        # RANDOM NOISE
        # ----------------------------------------------------

        "11": {
            "inputs": {
                "noise_seed": int(seed)
            },
            "class_type": "RandomNoise"
        },

        # ----------------------------------------------------
        # VIDEO VAE
        # ----------------------------------------------------

        "12": {
            "inputs": {
                "samples": [
                    "29",
                    0
                ],
                "vae": [
                    "1",
                    2
                ]
            },
            "class_type": "VAEDecode"
        },

        # ----------------------------------------------------
        # AUDIO VAE
        # ----------------------------------------------------

        "13": {
            "inputs": {
                "ckpt_name":
                    "ltx-av-step-1751000_vocoder_24K.safetensors"
            },
            "class_type": "LTXVAudioVAELoader"
        },

        "14": {
            "inputs": {
                "samples": [
                    "29",
                    1
                ],
                "audio_vae": [
                    "13",
                    0
                ]
            },
            "class_type": "LTXVAudioVAEDecode"
        },

        # ----------------------------------------------------
        # VIDEO OUTPUT
        # ----------------------------------------------------

        "15": {
            "inputs": {
                "frame_rate": [
                    "23",
                    0
                ],
                "loop_count": 0,
                "filename_prefix": "LTX2_5",
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "trim_to_audio": False,
                "pingpong": False,
                "save_output": True,
                "images": [
                    "12",
                    0
                ],
                "audio": [
                    "14",
                    0
                ]
            },
            "class_type": "VHS_VideoCombine"
        },

        # ----------------------------------------------------
        # MULTIMODAL GUIDER
        # ----------------------------------------------------

        "17": {
            "inputs": {
                "skip_blocks": "29",
                "model": [
                    "28",
                    1
                ],
                "positive": [
                    "22",
                    0
                ],
                "negative": [
                    "22",
                    1
                ],
                "parameters": [
                    "18",
                    0
                ]
            },
            "class_type": "MultimodalGuider"
        },

        # ----------------------------------------------------
        # VIDEO GUIDER PARAMETERS
        # ----------------------------------------------------

        "18": {
            "inputs": {
                "modality": "VIDEO",
                "cfg": 3,
                "stg": 0,
                "rescale": 0,
                "modality_scale": 3,
                "parameters": [
                    "19",
                    0
                ]
            },
            "class_type": "GuiderParameters"
        },

        # ----------------------------------------------------
        # AUDIO GUIDER PARAMETERS
        # ----------------------------------------------------

        "19": {
            "inputs": {
                "modality": "AUDIO",
                "cfg": 7,
                "stg": 0,
                "rescale": 0,
                "modality_scale": 3
            },
            "class_type": "GuiderParameters"
        },

        # ----------------------------------------------------
        # FRAME RATE
        # ----------------------------------------------------

        "23": {
            "inputs": {
                "value": float(fps)
            },
            "class_type": "FloatConstant"
        },

        "42": {
            "inputs": {
                "a": [
                    "23",
                    0
                ]
            },
            "class_type": "CM_FloatToInt"
        },

        # ----------------------------------------------------
        # AUDIO LATENT
        # ----------------------------------------------------

        "26": {
            "inputs": {
                "frames_number": [
                    "27",
                    0
                ],
                "frame_rate": [
                    "42",
                    0
                ],
                "batch_size": 1
            },
            "class_type": "LTXVEmptyLatentAudio"
        },

        "27": {
            "inputs": {
                "value": frames
            },
            "class_type": "INTConstant"
        },

        # ----------------------------------------------------
        # VIDEO LATENT
        # ----------------------------------------------------

        "43": {
            "inputs": {
                "width": int(width),
                "height": int(height),
                "length": [
                    "27",
                    0
                ],
                "batch_size": 1
            },
            "class_type": "EmptyLTXVLatentVideo"
        },

        # ----------------------------------------------------
        # CONCAT VIDEO + AUDIO
        # ----------------------------------------------------

        "28": {
            "inputs": {
                "video_latent": [
                    "43",
                    0
                ],
                "audio_latent": [
                    "26",
                    0
                ],
                "model": [
                    "44",
                    0
                ]
            },
            "class_type": "LTXVConcatAVLatent"
        },

        # ----------------------------------------------------
        # SEPARATE VIDEO + AUDIO
        # ----------------------------------------------------

        "29": {
            "inputs": {
                "av_latent": [
                    "41",
                    0
                ],
                "model": [
                    "28",
                    1
                ]
            },
            "class_type": "LTXVSeparateAVLatent"
        },

        # ----------------------------------------------------
        # SAMPLER
        # ----------------------------------------------------

        "41": {
            "inputs": {
                "noise": [
                    "11",
                    0
                ],
                "guider": [
                    "17",
                    0
                ],
                "sampler": [
                    "8",
                    0
                ],
                "sigmas": [
                    "9",
                    0
                ],
                "latent_image": [
                    "28",
                    0
                ]
            },
            "class_type": "SamplerCustomAdvanced"
        },

        # ----------------------------------------------------
        # MODEL PATCHER
        # ----------------------------------------------------

        "44": {
            "inputs": {
                "torch_compile": True,
                "disable_backup": False,
                "model": [
                    "1",
                    0
                ]
            },
            "class_type": "LTXVSequenceParallelMultiGPUPatcher"
        },

        # ----------------------------------------------------
        # CONDITIONING
        # ----------------------------------------------------

        "22": {
            "inputs": {
                "frame_rate": [
                    "23",
                    0
                ],
                "positive": [
                    "3",
                    0
                ],
                "negative": [
                    "4",
                    0
                ]
            },
            "class_type": "LTXVConditioning"
        }
    }

    # --------------------------------------------------------
    # Prompt enhancer
    #
    # The supplied workflow exposes prompt enhancement at the
    # high-level LTX-2.5 node. If your installed LTX nodes expose
    # a specific enhancer node, this can be connected there.
    # --------------------------------------------------------

    return workflow


# ============================================================
# COMFYUI API
# ============================================================

def queue_prompt(workflow):

    payload = {
        "prompt": workflow,
        "client_id": CLIENT_ID
    }

    response = requests.post(
        COMFYUI_URL + "/prompt",
        json=payload,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise RuntimeError(
            json.dumps(data, indent=2)
        )

    return data["prompt_id"]


def get_history(prompt_id):

    response = requests.get(
        COMFYUI_URL + f"/history/{prompt_id}",
        timeout=30
    )

    response.raise_for_status()

    return response.json()


def wait_for_generation(prompt_id):

    ws_url = (
        COMFYUI_URL
        .replace("http://", "ws://")
        .replace("https://", "wss://")
        + f"/ws?clientId={CLIENT_ID}"
    )

    ws = websocket.create_connection(
        ws_url,
        timeout=5
    )

    try:

        while True:

            try:
                message = ws.recv()

            except websocket.WebSocketTimeoutException:

                history = get_history(prompt_id)

                if prompt_id in history:

                    return history[prompt_id]

                continue

            if not message:
                continue

            if isinstance(message, bytes):
                continue

            data = json.loads(message)

            msg_type = data.get("type")

            msg_data = data.get("data", {})

            if msg_type == "executing":

                current_prompt = msg_data.get(
                    "prompt_id"
                )

                node = msg_data.get("node")

                if current_prompt == prompt_id:

                    if node is None:

                        time.sleep(1)

                        history = get_history(
                            prompt_id
                        )

                        if prompt_id in history:
                            return history[prompt_id]

            elif msg_type == "execution_error":

                if msg_data.get(
                    "prompt_id"
                ) == prompt_id:

                    raise RuntimeError(
                        json.dumps(
                            msg_data,
                            indent=2
                        )
                    )

    finally:

        ws.close()


# ============================================================
# FIND GENERATED VIDEO
# ============================================================

def find_video(history):

    outputs = history.get(
        "outputs",
        {}
    )

    for node_id, node_output in outputs.items():

        # VHS_VideoCombine normally exposes files
        # through this structure.

        videos = node_output.get(
            "gifs",
            []
        )

        for video in videos:

            if video.get("filename"):

                return video

        # Some versions use "videos"
        videos = node_output.get(
            "videos",
            []
        )

        for video in videos:

            if video.get("filename"):

                return video

    return None


# ============================================================
# DOWNLOAD VIDEO
# ============================================================

def download_video(video):

    filename = video["filename"]

    subfolder = video.get(
        "subfolder",
        ""
    )

    folder_type = video.get(
        "type",
        "output"
    )

    params = {
        "filename": filename,
        "subfolder": subfolder,
        "type": folder_type
    }

    response = requests.get(
        COMFYUI_URL + "/view",
        params=params,
        timeout=120
    )

    response.raise_for_status()

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    output_path = os.path.join(
        "outputs",
        filename
    )

    # Avoid accidentally creating nested
    # directories from subfolder names.

    output_path = os.path.abspath(
        output_path
    )

    with open(
        output_path,
        "wb"
    ) as f:

        f.write(
            response.content
        )

    return output_path


# ============================================================
# GENERATE
# ============================================================

def generate_video(
    prompt,
    negative_prompt,
    prompt_enhance,
    duration,
    width,
    height,
    fps,
    seed
):

    if not prompt or not prompt.strip():

        raise gr.Error(
            "Please enter a prompt."
        )

    try:

        width = int(width)
        height = int(height)
        fps = int(fps)
        duration = float(duration)

        if seed is None or int(seed) < 0:

            seed = random.randint(
                0,
                2**63 - 1
            )

        else:

            seed = int(seed)

        if width < 64 or height < 64:

            raise ValueError(
                "Width and height must be at least 64."
            )

        if duration <= 0:

            raise ValueError(
                "Duration must be greater than 0."
            )

        if fps <= 0:

            raise ValueError(
                "FPS must be greater than 0."
            )

        # ----------------------------------------------------
        # Prompt enhancement
        # ----------------------------------------------------
        #
        # The supplied LTX-2.5 workflow exposes this as a
        # workflow-level option.
        #
        # We keep the UI option here. The actual enhancer must
        # be connected to the exact LTX node version installed
        # in ComfyUI.
        #
        final_prompt = prompt.strip()

        workflow = build_workflow(
            prompt=final_prompt,
            negative_prompt=(
                negative_prompt.strip()
                if negative_prompt
                else DEFAULT_NEGATIVE
            ),
            width=width,
            height=height,
            duration=duration,
            fps=fps,
            seed=seed,
            prompt_enhance=prompt_enhance
        )

        print(
            "\n========================================"
        )
        print(
            "           LTX-2.5 GENERATION"
        )
        print(
            "========================================"
        )

        print(
            f"Prompt: {final_prompt}"
        )

        print(
            f"Negative: {negative_prompt}"
        )

        print(
            f"Resolution: {width}x{height}"
        )

        print(
            f"Duration: {duration}s"
        )

        print(
            f"FPS: {fps}"
        )

        print(
            f"Seed: {seed}"
        )

        print(
            f"Prompt Enhance: {prompt_enhance}"
        )

        # ----------------------------------------------------
        # Queue
        # ----------------------------------------------------

        print(
            "\nQueueing workflow..."
        )

        prompt_id = queue_prompt(
            workflow
        )

        print(
            f"Prompt ID: {prompt_id}"
        )

        # ----------------------------------------------------
        # Wait
        # ----------------------------------------------------

        print(
            "Waiting for ComfyUI..."
        )

        history = wait_for_generation(
            prompt_id
        )

        # ----------------------------------------------------
        # Find video
        # ----------------------------------------------------

        video = find_video(
            history
        )

        if video is None:

            raise RuntimeError(
                "Generation finished, but no video "
                "was found in the ComfyUI output."
            )

        # ----------------------------------------------------
        # Download
        # ----------------------------------------------------

        print(
            "Downloading generated video..."
        )

        output_path = download_video(
            video
        )

        print(
            f"Saved: {output_path}"
        )

        return (
            output_path,
            f"Generation complete!\n\n"
            f"Seed: {seed}\n"
            f"Resolution: {width}x{height}\n"
            f"FPS: {fps}\n"
            f"Duration: {duration}s"
        )

    except requests.exceptions.ConnectionError:

        raise gr.Error(
            "Cannot connect to ComfyUI.\n\n"
            "Make sure ComfyUI is running on:\n"
            f"{COMFYUI_URL}"
        )

    except Exception as e:

        print(
            "\nERROR:"
        )

        print(
            str(e)
        )

        raise gr.Error(
            str(e)
        )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="LTX-2.5 Video Generator"
) as app:

    gr.Markdown(
        """
# 🎬 LTX-2.5 Video Generator

Generate videos through your **ComfyUI LTX-2.5 workflow**.

**No LoRAs.**
"""
    )

    with gr.Row():

        with gr.Column(
            scale=2
        ):

            prompt = gr.Textbox(
                label="Positive Prompt",
                placeholder=(
                    "Describe the video you want..."
                ),
                lines=6
            )

            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                value=DEFAULT_NEGATIVE,
                lines=4
            )

            prompt_enhance = gr.Checkbox(
                label="Prompt Enhance",
                value=True
            )

            with gr.Row():

                duration = gr.Number(
                    label="Duration (seconds)",
                    value=DEFAULT_DURATION,
                    minimum=1,
                    maximum=60,
                    step=1
                )

                fps = gr.Number(
                    label="FPS",
                    value=DEFAULT_FPS,
                    minimum=1,
                    maximum=60,
                    step=1
                )

            with gr.Row():

                width = gr.Number(
                    label="Width",
                    value=DEFAULT_WIDTH,
                    minimum=64,
                    step=64
                )

                height = gr.Number(
                    label="Height",
                    value=DEFAULT_HEIGHT,
                    minimum=64,
                    step=64
                )

            seed = gr.Number(
                label="Seed (-1 = Random)",
                value=-1,
                precision=0
            )

            generate_button = gr.Button(
                "🚀 Generate Video",
                variant="primary",
                size="lg"
            )

        with gr.Column(
            scale=2
        ):

            video_output = gr.Video(
                label="Generated Video",
                autoplay=False
            )

            status = gr.Textbox(
                label="Status",
                lines=6
            )

    generate_button.click(
        fn=generate_video,
        inputs=[
            prompt,
            negative_prompt,
            prompt_enhance,
            duration,
            width,
            height,
            fps,
            seed
        ],
        outputs=[
            video_output,
            status
        ]
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    print(
        "\n=========================================="
    )
    print(
        "       LTX-2.5 COMFYUI VIDEO APP"
    )
    print(
        "=========================================="
    )

    print(
        f"ComfyUI: {COMFYUI_URL}"
    )

    print(
        "LoRAs: DISABLED"
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
