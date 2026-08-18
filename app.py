# ============================================================
# LTX-2.5 DIRECT COMFYUI NODE APP
#
# Same architecture as the user's working Krea-2 app:
#
#     Gradio
#        ↓
# NODE_CLASS_MAPPINGS
#        ↓
# LTX-2.5 subgraph
#        ↓
# Video
#
# NO ComfyUI HTTP SERVER
# NO port 8188
# NO WebSocket
# NO LoRAs
# ============================================================

import os
import sys
import time
import random
import inspect
import traceback
from pathlib import Path

import torch
import gradio as gr

from nodes import NODE_CLASS_MAPPINGS


# ============================================================
# SETTINGS
# ============================================================

APP_TITLE = "LTX-2.5 Video Generator"

COMFYUI_ROOT = Path(
    os.environ.get(
        "COMFYUI_ROOT",
        "/root/ComfyUI"
    )
)

OUTPUT_DIR = COMFYUI_ROOT / "output"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL NAMES FROM THE EXACT WORKFLOW
# ============================================================

LTX_MODEL = (
    "ltx-2.5-22b-distilled-transformer-"
    "comfy-int8-convrot.safetensors"
)

VIDEO_VAE = (
    "ltx-2.5-video-vae-bf16.safetensors"
)

AUDIO_VAE = (
    "ltx-2.5-audio-vae-bf16.safetensors"
)

UPSCALE_MODEL = (
    "ltx-2.5-latent-spatial-upscaler-x2-"
    "bf16-1.0.safetensors"
)

PROMPT_ENHANCE_MODEL = (
    "gemma4-12b-with-proj-ltx-2.5-"
    "comfy-int8-convrot.safetensors"
)


DEFAULT_NEGATIVE = (
    "blurry, low quality, still frame, frames, "
    "watermark, overlay, titles, has blurbox, "
    "has subtitles"
)


# ============================================================
# FIND THE LTX-2.5 SUBGRAPH
# ============================================================

def find_ltx25_node():

    print("\n" + "=" * 60)
    print("Searching for LTX-2.5 ComfyUI node...")
    print("=" * 60)

    # --------------------------------------------------------
    # The workflow supplied by the user contains this UUID as
    # the subgraph type.
    # --------------------------------------------------------

    known_uuid = (
        "8b4f085c-1bb3-4ecd-aeed-603a8d6d3970"
    )

    if known_uuid in NODE_CLASS_MAPPINGS:

        cls = NODE_CLASS_MAPPINGS[
            known_uuid
        ]

        print(
            f"✓ Found LTX-2.5 subgraph: {known_uuid}"
        )

        return cls

    # --------------------------------------------------------
    # Try to find it by its input signature.
    #
    # This makes the app more tolerant of ComfyUI versions
    # that register the subgraph under a different internal key.
    # --------------------------------------------------------

    wanted = {
        "text",
        "value",
        "value_1",
        "value_2",
        "value_3",
        "noise_seed",
        "value_4",
        "vae_name",
        "vae_name_1",
        "model_name",
        "clip_name_1",
    }

    candidates = []

    for key, cls in NODE_CLASS_MAPPINGS.items():

        try:

            if not hasattr(
                cls,
                "INPUT_TYPES"
            ):
                continue

            info = cls.INPUT_TYPES()

            required = info.get(
                "required",
                {}
            )

            optional = info.get(
                "optional",
                {}
            )

            names = (
                set(required.keys())
                |
                set(optional.keys())
            )

            score = len(
                wanted.intersection(names)
            )

            if score >= 6:

                candidates.append(
                    (
                        score,
                        key,
                        cls,
                        names
                    )
                )

        except Exception:
            continue

    candidates.sort(
        reverse=True,
        key=lambda x: x[0]
    )

    if candidates:

        score, key, cls, names = candidates[0]

        print(
            f"✓ Found probable LTX-2.5 node: {key}"
        )

        print(
            f"  Signature score: {score}"
        )

        return cls

    # --------------------------------------------------------
    # Print useful diagnostics
    # --------------------------------------------------------

    print(
        "\n❌ Could not find the LTX-2.5 subgraph."
    )

    print(
        "\nAvailable LTX-related nodes:"
    )

    for key in NODE_CLASS_MAPPINGS:

        name = str(key).lower()

        if "ltx" in name:

            print(
                " ",
                key
            )

    raise RuntimeError(
        "LTX-2.5 ComfyUI node was not found.\n\n"
        "Make sure your ComfyUI version contains "
        "the LTX-2.5 workflow nodes."
    )


LTX25_NODE_CLASS = find_ltx25_node()


# ============================================================
# DISPLAY NODE INPUTS
# ============================================================

try:

    LTX_INPUT_TYPES = (
        LTX25_NODE_CLASS.INPUT_TYPES()
    )

    print(
        "\nLTX-2.5 node inputs:"
    )

    for section in (
        "required",
        "optional"
    ):

        for name in LTX_INPUT_TYPES.get(
            section,
            {}
        ):

            print(
                f"  {name}"
            )

except Exception as e:

    print(
        "Could not inspect LTX node:",
        e
    )


# ============================================================
# NODE INSTANCE
# ============================================================

print(
    "\nInitializing LTX-2.5 node..."
)

LTX25_NODE = LTX25_NODE_CLASS()


print(
    "✓ LTX-2.5 node initialized"
)


# ============================================================
# HELPERS
# ============================================================

def get_function_name(node):

    function_name = getattr(
        node,
        "FUNCTION",
        None
    )

    if function_name:
        return function_name

    # Fallbacks used by various ComfyUI nodes
    for name in (
        "execute",
        "generate",
        "run",
        "process",
    ):

        if hasattr(
            node,
            name
        ):

            return name

    raise RuntimeError(
        "Could not determine the LTX-2.5 "
        "node execution function."
    )


def get_callable_inputs(node):

    function_name = get_function_name(
        node
    )

    function = getattr(
        node,
        function_name
    )

    try:

        signature = inspect.signature(
            function
        )

        return signature

    except Exception:

        return None


def make_frame_count(
    duration,
    fps
):

    # LTX video latent length follows the
    # 8n + 1 pattern.

    frames = int(
        round(
            float(duration)
            * int(fps)
        )
    )

    frames = max(
        9,
        frames
    )

    remainder = (
        frames - 1
    ) % 8

    if remainder:

        frames += (
            8 - remainder
        )

    return frames


# ============================================================
# EXECUTE LTX SUBGRAPH
# ============================================================

def execute_ltx25(
    prompt,
    prompt_enhance,
    duration,
    width,
    height,
    seed,
    fps
):

    duration = int(
        duration
    )

    width = int(
        width
    )

    height = int(
        height
    )

    fps = int(
        fps
    )

    seed = int(
        seed
    )

    # --------------------------------------------------------
    # LTX resolution
    # --------------------------------------------------------

    width = max(
        32,
        (width // 32) * 32
    )

    height = max(
        32,
        (height // 32) * 32
    )

    # --------------------------------------------------------
    # LTX frame count
    # --------------------------------------------------------

    frames = make_frame_count(
        duration,
        fps
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "LTX-2.5 GENERATION"
    )

    print(
        "=" * 60
    )

    print(
        f"Prompt: {prompt}"
    )

    print(
        f"Prompt Enhance: {prompt_enhance}"
    )

    print(
        f"Duration: {duration}s"
    )

    print(
        f"Resolution: {width}x{height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"Frames: {frames}"
    )

    print(
        f"Seed: {seed}"
    )

    # --------------------------------------------------------
    # Inspect actual node signature
    # --------------------------------------------------------

    input_types = (
        LTX25_NODE_CLASS.INPUT_TYPES()
    )

    required = input_types.get(
        "required",
        {}
    )

    optional = input_types.get(
        "optional",
        {}
    )

    available = (
        set(required.keys())
        |
        set(optional.keys())
    )

    # --------------------------------------------------------
    # Build exact workflow inputs
    #
    # These names correspond to the inputs exposed by the
    # supplied LTX-2.5 Text-to-Video subgraph.
    # --------------------------------------------------------

    values = {}

    # Prompt
    if "text" in available:

        values["text"] = prompt

    elif "prompt" in available:

        values["prompt"] = prompt

    # Prompt enhancer
    if "value" in available:

        values["value"] = bool(
            prompt_enhance
        )

    elif "prompt_enhance" in available:

        values["prompt_enhance"] = bool(
            prompt_enhance
        )

    # Duration
    if "value_1" in available:

        values["value_1"] = duration

    elif "duration" in available:

        values["duration"] = duration

    # Width
    if "value_2" in available:

        values["value_2"] = width

    elif "width" in available:

        values["width"] = width

    # Height
    if "value_3" in available:

        values["value_3"] = height

    elif "height" in available:

        values["height"] = height

    # Seed
    if "noise_seed" in available:

        values["noise_seed"] = seed

    elif "seed" in available:

        values["seed"] = seed

    # FPS
    if "value_4" in available:

        values["value_4"] = fps

    elif "frame_rate" in available:

        values["frame_rate"] = fps

    # Video VAE
    if "vae_name" in available:

        values["vae_name"] = VIDEO_VAE

    elif "video_vae" in available:

        values["video_vae"] = VIDEO_VAE

    # Audio VAE
    if "vae_name_1" in available:

        values["vae_name_1"] = AUDIO_VAE

    elif "audio_vae" in available:

        values["audio_vae"] = AUDIO_VAE

    # Spatial latent upscaler
    if "model_name" in available:

        values["model_name"] = UPSCALE_MODEL

    elif "upscale_model" in available:

        values["upscale_model"] = UPSCALE_MODEL

    # Prompt enhancer model
    if "clip_name_1" in available:

        values["clip_name_1"] = (
            PROMPT_ENHANCE_MODEL
        )

    elif "prompt_enhance_model" in available:

        values[
            "prompt_enhance_model"
        ] = PROMPT_ENHANCE_MODEL

    # --------------------------------------------------------
    # Filter only arguments accepted by the actual function.
    # --------------------------------------------------------

    signature = get_callable_inputs(
        LTX25_NODE
    )

    if signature:

        parameters = signature.parameters

        filtered = {}

        for name, value in values.items():

            if name in parameters:

                filtered[name] = value

        values = filtered

    # --------------------------------------------------------
    # Show arguments
    # --------------------------------------------------------

    print(
        "\nCalling LTX-2.5 node with:"
    )

    for key, value in values.items():

        print(
            f"  {key}: {value}"
        )

    # --------------------------------------------------------
    # Execute directly
    # --------------------------------------------------------

    function_name = (
        get_function_name(
            LTX25_NODE
        )
    )

    function = getattr(
        LTX25_NODE,
        function_name
    )

    print(
        f"\nExecuting node function: "
        f"{function_name}"
    )

    with torch.inference_mode():

        result = function(
            **values
        )

    print(
        "\n✓ LTX-2.5 generation finished"
    )

    return result


# ============================================================
# EXTRACT VIDEO FROM NODE RESULT
# ============================================================

def extract_video(result):

    if result is None:

        return None

    # --------------------------------------------------------
    # ComfyUI nodes normally return tuples.
    # --------------------------------------------------------

    if isinstance(
        result,
        tuple
    ):

        for item in result:

            video = extract_video(
                item
            )

            if video is not None:

                return video

        return None

    # --------------------------------------------------------
    # List
    # --------------------------------------------------------

    if isinstance(
        result,
        list
    ):

        for item in result:

            video = extract_video(
                item
            )

            if video is not None:

                return video

        return None

    # --------------------------------------------------------
    # Dictionary
    # --------------------------------------------------------

    if isinstance(
        result,
        dict
    ):

        # Direct VIDEO
        if "video" in result:

            return result[
                "video"
            ]

        if "VIDEO" in result:

            return result[
                "VIDEO"
            ]

        # Search recursively
        for value in result.values():

            video = extract_video(
                value
            )

            if video is not None:

                return video

        return None

    # --------------------------------------------------------
    # Object
    # --------------------------------------------------------

    class_name = (
        result.__class__.__name__
    ).lower()

    if (
        "video" in class_name
        or "av" in class_name
    ):

        return result

    return None


# ============================================================
# SAVE VIDEO DIRECTLY THROUGH COMFYUI NODE
# ============================================================

def save_video_direct(
    video,
    seed
):

    if video is None:

        raise RuntimeError(
            "LTX-2.5 returned no VIDEO output."
        )

    if "SaveVideo" not in (
        NODE_CLASS_MAPPINGS
    ):

        raise RuntimeError(
            "SaveVideo node is not available "
            "in this ComfyUI installation."
        )

    SaveVideoClass = (
        NODE_CLASS_MAPPINGS[
            "SaveVideo"
        ]
    )

    saver = SaveVideoClass()

    function_name = (
        get_function_name(
            saver
        )
    )

    function = getattr(
        saver,
        function_name
    )

    # --------------------------------------------------------
    # Inspect SaveVideo's actual API
    # --------------------------------------------------------

    input_types = (
        SaveVideoClass.INPUT_TYPES()
    )

    available = set()

    for section in (
        "required",
        "optional"
    ):

        available.update(
            input_types.get(
                section,
                {}
            ).keys()
        )

    values = {}

    if "video" in available:

        values[
            "video"
        ] = video

    if "filename_prefix" in available:

        values[
            "filename_prefix"
        ] = "LTX2.5/video"

    if "format" in available:

        values[
            "format"
        ] = "auto"

    if "codec" in available:

        values[
            "codec"
        ] = "auto"

    if "filename_prefix" in available:

        print(
            "\nSaving video..."
        )

    signature = get_callable_inputs(
        saver
    )

    if signature:

        parameters = signature.parameters

        values = {
            k: v
            for k, v in values.items()
            if k in parameters
        }

    result = function(
        **values
    )

    return result


# ============================================================
# FIND NEW MP4
# ============================================================

def find_new_video(
    before_files,
    started_at
):

    candidates = []

    for path in OUTPUT_DIR.rglob(
        "*.mp4"
    ):

        try:

            stat = path.stat()

            if (
                path not in before_files
                and stat.st_mtime >= started_at
            ):

                candidates.append(
                    path
                )

        except Exception:

            pass

    if not candidates:

        # Sometimes the saver reuses a filename.
        for path in OUTPUT_DIR.rglob(
            "*.mp4"
        ):

            try:

                if path.stat().st_mtime >= (
                    started_at - 2
                ):

                    candidates.append(
                        path
                    )

            except Exception:

                pass

    if not candidates:

        return None

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return candidates[0]


# ============================================================
# GENERATE VIDEO
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

    # --------------------------------------------------------
    # The LTX-2.5 workflow supplied by the user exposes the
    # negative prompt internally.
    #
    # Keep this UI field for compatibility, but do not inject
    # it into an unrelated node. The exact supplied workflow
    # uses its own negative-conditioning branch.
    # --------------------------------------------------------

    if not prompt or not prompt.strip():

        raise gr.Error(
            "Please enter a prompt."
        )

    try:

        # Random seed
        if (
            seed is None
            or int(seed) < 0
        ):

            seed = random.randint(
                0,
                2**63 - 1
            )

        else:

            seed = int(seed)

        duration = int(
            duration
        )

        width = int(
            width
        )

        height = int(
            height
        )

        fps = int(
            fps
        )

        # ----------------------------------------------------
        # Capture existing outputs
        # ----------------------------------------------------

        before_files = set(
            OUTPUT_DIR.rglob(
                "*.mp4"
            )
        )

        started_at = time.time()

        # ----------------------------------------------------
        # Execute exact LTX-2.5 subgraph
        # ----------------------------------------------------

        result = execute_ltx25(
            prompt=prompt.strip(),
            prompt_enhance=bool(
                prompt_enhance
            ),
            duration=duration,
            width=width,
            height=height,
            seed=seed,
            fps=fps
        )

        # ----------------------------------------------------
        # Try to extract VIDEO
        # ----------------------------------------------------

        video = extract_video(
            result
        )

        # ----------------------------------------------------
        # Save using ComfyUI SaveVideo
        # ----------------------------------------------------

        if video is not None:

            try:

                save_video_direct(
                    video,
                    seed
                )

            except Exception as save_error:

                print(
                    "\nSaveVideo node warning:"
                )

                print(
                    save_error
                )

        # ----------------------------------------------------
        # Locate generated MP4
        # ----------------------------------------------------

        output_path = None

        for _ in range(30):

            output_path = (
                find_new_video(
                    before_files,
                    started_at
                )
            )

            if output_path:

                break

            time.sleep(
                1
            )

        if output_path is None:

            raise RuntimeError(
                "Generation completed but "
                "no MP4 was found in:\n"
                f"{OUTPUT_DIR}"
            )

        print(
            "\n" + "=" * 60
        )

        print(
            "VIDEO SAVED"
        )

        print(
            output_path
        )

        print(
            "=" * 60
        )

        return (
            str(output_path),
            (
                "✅ Generation complete\n\n"
                f"Seed: {seed}\n"
                f"Resolution: "
                f"{width} × {height}\n"
                f"Duration: {duration}s\n"
                f"FPS: {fps}\n\n"
                f"Saved:\n{output_path}"
            )
        )

    except Exception as e:

        traceback.print_exc()

        raise gr.Error(
            str(e)
        )


# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title=APP_TITLE
) as app:

    gr.Markdown(
        """
# 🎬 LTX-2.5 Video Generator

Direct LTX-2.5 ComfyUI node execution.

**No LoRAs • No ComfyUI server • No API**
"""
    )

    with gr.Row():

        with gr.Column(
            scale=2
        ):

            prompt = gr.Textbox(
                label="Positive Prompt",
                placeholder=(
                    "Describe the video..."
                ),
                lines=7
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
                    value=5,
                    minimum=1,
                    maximum=60,
                    step=1
                )

                fps = gr.Number(
                    label="FPS",
                    value=24,
                    minimum=1,
                    maximum=60,
                    step=1
                )

            with gr.Row():

                width = gr.Number(
                    label="Width",
                    value=1280,
                    minimum=32,
                    step=32
                )

                height = gr.Number(
                    label="Height",
                    value=720,
                    minimum=32,
                    step=32
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
                lines=8
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
# START APP
# ============================================================

if __name__ == "__main__":

    print(
        "\n" + "=" * 60
    )

    print(
        "        LTX-2.5 DIRECT NODE APP"
    )

    print(
        "=" * 60
    )

    print(
        "ComfyUI API: DISABLED"
    )

    print(
        "ComfyUI port 8188: NOT REQUIRED"
    )

    print(
        "LoRAs: DISABLED"
    )

    print(
        f"Output: {OUTPUT_DIR}"
    )

    print(
        "=" * 60
    )

    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )
