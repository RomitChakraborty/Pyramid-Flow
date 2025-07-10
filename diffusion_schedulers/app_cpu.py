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

# Force CPU-only, full precision
DEVICE = torch.device('cpu')
print(f"[INFO] Using device: {DEVICE}")

def download_model(repo_id: str, local_dir: str):
    """Download model from Hugging Face if not already present."""
    if not os.path.isdir(local_dir) or not os.listdir(local_dir):
        print(f"[INFO] Downloading model {repo_id} to {local_dir}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            repo_type='model'
        )
        print("[INFO] Download complete.")
    else:
        print("[INFO] Model already present; skipping download.")

# Configuration
MODEL_NAME = "pyramid_flux"
MODEL_REPO = "rain1011/pyramid-flow-miniflux"
VARIANTS = {'384p': 'diffusion_transformer_384p', '768p': 'diffusion_transformer_768p'}
MODEL_DIR = os.path.join(os.getcwd(), 'pyramid_flow_model')

# Thread-safe cache
_model_cache = {}
_cache_lock = threading.Lock()

# Download model once
download_model(MODEL_REPO, MODEL_DIR)

# Initialize and cache pipeline

def get_pipeline(resolution: str):
    """Load or retrieve a cached CPU pipeline for the given resolution."""
    with _cache_lock:
        if resolution in _model_cache:
            return _model_cache[resolution]
        print(f"[INFO] Initializing {resolution} pipeline...")
        variant = VARIANTS[resolution]
        pipeline = PyramidDiTForVideoGeneration(
            base_path=MODEL_DIR,
            model_name=MODEL_NAME,
            model_dtype='fp32',
            model_variant=variant,
            cpu_offloading=False
        )
        pipeline.vae.enable_tiling()
        # Move all modules to CPU
        pipeline.vae.to(DEVICE, dtype=torch.float32)
        pipeline.dit.to(DEVICE, dtype=torch.float32)
        pipeline.text_encoder.to(DEVICE, dtype=torch.float32)
        _model_cache[resolution] = pipeline
        print("[INFO] Pipeline ready.")
        return pipeline

# Image preprocessing
def preprocess_image(img: Image.Image, resolution: str):
    """Resize and center-crop image to model dimensions."""
    if resolution == '768p':
        w, h = 1280, 768
    else:
        w, h = 640, 384
    ow, oh = img.size
    scale = max(w / ow, h / oh)
    nw, nh = int(ow * scale), int(oh * scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    return img.crop((left, top, left + w, top + h))

# Generation functions

def generate_text_to_video(prompt, duration, guidance, vguidance, resolution, seed, progress=gr.Progress()):
    print(f"[INFO] Text2Video: '{prompt}' duration={duration}, guidance={guidance}, vguidance={vguidance}")
    pipeline = get_pipeline(resolution)
    steps = min(max(int(duration), 1), 8)
    height, width = (768, 1280) if resolution == '768p' else (384, 640)
    def cb(i, total):
        progress(i / total)
        print(f"[INFO] Step {i}/{int(total)}")
    with torch.no_grad():
        frames = pipeline.generate(
            prompt=prompt,
            num_inference_steps=[steps] * 3,
            video_num_inference_steps=[max(1, steps // 2)] * 3,
            height=height,
            width=width,
            temp=duration,
            guidance_scale=guidance,
            video_guidance_scale=vguidance,
            output_type='pil',
            cpu_offloading=False,
            save_memory=True,
            callback=cb
        )
    out_path = f"{uuid.uuid4()}_text2video.mp4"
    export_to_video(frames, out_path, fps=12)
    print(f"[INFO] Video saved: {out_path}")
    return out_path


def generate_image_to_video(image, prompt, duration, vguidance, resolution, seed, progress=gr.Progress()):
    print(f"[INFO] Img2Video: '{prompt}' duration={duration}, vguidance={vguidance}")
    img = preprocess_image(image, resolution)
    pipeline = get_pipeline(resolution)
    steps = min(max(int(duration), 1), 8)
    def cb(i, total):
        progress(i / total)
        print(f"[INFO] Step {i}/{int(total)}")
    with torch.no_grad():
        frames = pipeline.generate_i2v(
            prompt=prompt,
            input_image=img,
            num_inference_steps=[steps] * 3,
            temp=duration,
            video_guidance_scale=vguidance,
            output_type='pil',
            cpu_offloading=False,
            save_memory=True,
            callback=cb
        )
    out_path = f"{uuid.uuid4()}_img2video.mp4"
    export_to_video(frames, out_path, fps=12)
    print(f"[INFO] Video saved: {out_path}")
    return out_path

# Gradio UI
def main():
    demo = gr.Blocks()
    with demo:
        gr.Markdown("# Pyramid Flow CPU Video Generation")
        res_dd = gr.Dropdown(['384p', '768p'], value='384p', label='Resolution')
        with gr.Tab('Text-to-Video'):
            tp = gr.Textbox(label='Prompt', lines=2)
            td = gr.Slider(1, 16, value=5, label='Duration')
            tg = gr.Slider(1, 15, value=7, label='Guidance Scale')
            tv = gr.Slider(1, 10, value=5, label='Video Guidance')
            ts = gr.Number(value=0, label='Seed')
            tb = gr.Button('Generate')
            tout = gr.Video()
            tb.click(generate_text_to_video, [tp, td, tg, tv, res_dd, ts], tout)
        with gr.Tab('Image-to-Video'):
            ipt = gr.Image(type='pil')
            ip = gr.Textbox(label='Prompt', lines=2)
            idur = gr.Slider(1, 16, value=5, label='Duration')
            iv = gr.Slider(1, 10, value=5, label='Video Guidance')
            iseed = gr.Number(value=0, label='Seed')
            ib = gr.Button('Generate')
            iout = gr.Video()
            ib.click(generate_image_to_video, [ipt, ip, idur, iv, res_dd, iseed], iout)
    demo.launch(share=False)

if __name__ == '__main__':
    main()
