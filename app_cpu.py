import multiprocessing as mp

# macOS default is 'spawn'; switch to 'fork' to avoid semaphore leaks
try:
    mp.set_start_method("fork")
except RuntimeError:
    pass

import os
import uuid
import threading
import random
import gradio as gr
import torch
from PIL import Image
from pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download

# -------------------------------------------------
# CPU-ONLY, FULL-PRECISION (FP32) IMPLEMENTATION
# -------------------------------------------------

DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.float32
print(f"[INFO] Using device: {DEVICE}, dtype: {TORCH_DTYPE}")

MODEL_NAME = "pyramid_flux"
MODEL_REPO = "rain1011/pyramid-flow-miniflux"
MODEL_DIR = os.path.join(os.getcwd(), "pyramid_flow_model")
VARIANTS = {"384p": "diffusion_transformer_384p", "768p": "diffusion_transformer_768p"}

_pipeline_cache = {}
_cache_lock = threading.Lock()

# Download once at startup
if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
    print(f"[INFO] Downloading model from {MODEL_REPO} into {MODEL_DIR}...")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=MODEL_DIR,
        local_dir_use_symlinks=False,
        repo_type="model",
    )
    print("[INFO] Model download complete.")
else:
    print("[INFO] Model already present; skipping download.")


def get_pipeline(resolution: str) -> PyramidDiTForVideoGeneration:
    with _cache_lock:
        if resolution in _pipeline_cache:
            return _pipeline_cache[resolution]

        print(f"[INFO] Initializing {resolution} pipeline on CPU...")
        variant = VARIANTS[resolution]
        pipe = PyramidDiTForVideoGeneration(
            MODEL_DIR,
            model_name=MODEL_NAME,
            model_dtype="fp32",
            model_variant=variant,
            cpu_offloading=False,
        )
        pipe.vae.enable_tiling()
        # move everything to CPU float32
        pipe.vae.to(DEVICE, dtype=TORCH_DTYPE)
        pipe.dit.to(DEVICE, dtype=TORCH_DTYPE)
        pipe.text_encoder.to(DEVICE, dtype=TORCH_DTYPE)

        _pipeline_cache[resolution] = pipe
        print(f"[INFO] {resolution} pipeline ready.")
        return pipe


def preprocess_image(img: Image.Image, resolution: str) -> Image.Image:
    w, h = (1280, 768) if resolution == "768p" else (640, 384)
    ow, oh = img.size
    scale = max(w / ow, h / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    resized = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return resized.crop((left, top, left + w, top + h))


def generate_text_to_video(
    prompt: str,
    duration: float,
    guidance: float,
    vguidance: float,
    resolution: str,
    seed: int,
    progress=gr.Progress(),
) -> str:
    try:
        print(
            f"[INFO] Text2Video: '{prompt}' dur={duration}, guide={guidance}, "
            f"vguide={vguidance}, res={resolution}, seed={seed}"
        )
        pipe = get_pipeline(resolution)
        steps = max(1, min(int(duration), 8))
        height, width = (768, 1280) if resolution == "768p" else (384, 640)

        def cb(i, total):
            progress(i / total)

        with torch.no_grad():
            frames = pipe.generate(
                prompt=prompt,
                num_inference_steps=[steps] * 3,
                video_num_inference_steps=[max(1, steps // 2)] * 3,
                height=height,
                width=width,
                temp=duration,
                guidance_scale=guidance,
                video_guidance_scale=vguidance,
                output_type="pil",
                cpu_offloading=False,
                save_memory=True,
                callback=cb,
            )

        out_path = f"{uuid.uuid4()}_text2vid.mp4"
        export_to_video(frames, out_path, fps=12)
        print(f"[INFO] Video saved: {out_path}")
        return out_path

    except Exception as e:
        print(f"[ERROR] Text2Video failed: {e}", flush=True)
        return f"Error: {e}"


def generate_image_to_video(
    image: Image.Image,
    prompt: str,
    duration: float,
    vguidance: float,
    resolution: str,
    seed: int,
    progress=gr.Progress(),
) -> str:
    try:
        print(
            f"[INFO] Img2Video: '{prompt}' dur={duration}, vguide={vguidance}, "
            f"res={resolution}, seed={seed}"
        )
        img = preprocess_image(image, resolution)
        pipe = get_pipeline(resolution)
        steps = max(1, min(int(duration), 8))

        def cb(i, total):
            progress(i / total)

        with torch.no_grad():
            frames = pipe.generate_i2v(
                prompt=prompt,
                input_image=img,
                num_inference_steps=[steps] * 3,
                temp=duration,
                video_guidance_scale=vguidance,
                output_type="pil",
                cpu_offloading=False,
                save_memory=True,
                callback=cb,
            )

        out_path = f"{uuid.uuid4()}_img2vid.mp4"
        export_to_video(frames, out_path, fps=12)
        print(f"[INFO] Video saved: {out_path}")
        return out_path

    except Exception as e:
        print(f"[ERROR] Img2Video failed: {e}", flush=True)
        return f"Error: {e}"


def main():
    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Pyramid Flow CPU Video Generation Demo")

        res_dd = gr.Dropdown(["384p", "768p"], value="384p", label="Resolution")

        with gr.Tab("Text-to-Video"):
            tp = gr.Textbox(label="Prompt", lines=2)
            td = gr.Slider(1, 16, value=5, label="Duration")
            tg = gr.Slider(1, 15, value=7, label="Guidance Scale")
            tv = gr.Slider(1, 10, value=5, label="Video Guidance")
            ts = gr.Number(value=0, label="Seed")
            tb = gr.Button("Generate")
            tout = gr.Video()

            tb.click(
                generate_text_to_video,
                [tp, td, tg, tv, res_dd, ts],
                tout,
                concurrency_limit=1,
            )

        with gr.Tab("Image-to-Video"):
            ip = gr.Image(type="pil", label="Input Image")
            ipp = gr.Textbox(label="Prompt", lines=2)
            idur = gr.Slider(1, 16, value=5, label="Duration")
            ivg = gr.Slider(1, 10, value=5, label="Video Guidance")
            iseed = gr.Number(value=0, label="Seed")
            ib = gr.Button("Generate")
            iout = gr.Video()

            ib.click(
                generate_image_to_video,
                [ip, ipp, idur, ivg, res_dd, iseed],
                iout,
                concurrency_limit=1,
            )

    demo.launch(share=False)


if __name__ == "__main__":
    main()
