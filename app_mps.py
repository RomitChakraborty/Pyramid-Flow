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

# --- Device configuration: CPU only to avoid MPS issues ---
global_device = torch.device("cpu")
print(f"[INFO] Using device: {global_device}")

# --- Model configuration ---
model_name = "pyramid_flux"
model_repo = (
    "rain1011/pyramid-flow-sd3" if model_name == "pyramid_mmdit"
    else "rain1011/pyramid-flow-miniflux"
)
variants = {'high': 'diffusion_transformer_768p', 'low': 'diffusion_transformer_384p'}
required_file = 'config.json'
width_high, height_high = 1280, 768
width_low, height_low = 640, 384

download_dir = os.path.join(os.getcwd(), "pyramid_flow_model")

# Thread-safe cache
model_cache = {}
model_cache_lock = threading.Lock()

def download_model(repo_id, local_dir, variants, required_file):
    """Download model if not already present"""
    need = not os.path.isdir(local_dir)
    if not need:
        for v in variants.values():
            if not os.path.exists(os.path.join(local_dir, v, required_file)):
                need = True
                break
    if need:
        print(f"[INFO] Downloading model '{repo_id}'...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            repo_type='model'
        )
        print("[INFO] Download complete.")
    else:
        print("[INFO] Model already present; skipping download.")


def initialize_model(variant):
    """Initialize specified variant on CPU in float32."""
    print(f"[INFO] Initializing model variant='{variant}'...")
    variant_dir = variants['high'] if variant == '768p' else variants['low']
    cfg_path = os.path.join(download_dir, variant_dir, 'config.json')
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"Missing config.json in {variant_dir}")
    
    model = PyramidDiTForVideoGeneration(
        download_dir,
        model_name=model_name,
        model_dtype='fp32',
        model_variant=variant_dir,
        cpu_offloading=False
    )
    model.vae.enable_tiling()
    # Move all modules to CPU with float32
    print("[INFO] Moving VAE, DiT, and text encoder to CPU (float32)...")
    model.vae.to(global_device, dtype=torch.float32)
    model.dit.to(global_device, dtype=torch.float32)
    model.text_encoder.to(global_device, dtype=torch.float32)
    print("[INFO] Model initialized on CPU.")
    return model, torch.float32


def initialize_model_cached(variant, seed):
    """Seed and cache model initialization."""
    if seed == 0:
        seed = random.randint(0, 2**8 - 1)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    with model_cache_lock:
        if variant not in model_cache:
            model_cache[variant] = initialize_model(variant)
    return model_cache[variant]


def preload_models():
    """Preload both variants at startup."""
    print("[INFO] Preloading model variants...")
    for var in ['384p', '768p']:
        initialize_model_cached(var, seed=42)
    print("[INFO] Preloading complete.")


def resize_crop_image(img: Image.Image, tgt_w, tgt_h):
    """Resize and center-crop the input PIL image."""
    ow, oh = img.size
    scale = max(tgt_w/ow, tgt_h/oh)
    nw, nh = round(ow*scale), round(oh*scale)
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tgt_w) / 2, (nh - tgt_h) / 2
    cropped = img.crop((left, top, left + tgt_w, top + tgt_h))
    return cropped


def generate_text_to_video(prompt, duration, guidance_scale, video_guidance_scale, resolution, seed, progress=gr.Progress()):
    print(f"[INFO] generate_text_to_video called with prompt='{prompt}'...")
    variant = '768p' if resolution == '768p' else '384p'
    h, w = (height_high, width_high) if variant == '768p' else (height_low, width_low)
    model, dtype = initialize_model_cached(variant, seed)
    print("[INFO] Starting text-to-video generation...")
    with torch.no_grad():
        frames = model.generate(
            prompt=prompt,
            num_inference_steps=[duration, duration, duration],
            video_num_inference_steps=[max(1, duration//2)]*3,
            height=h,
            width=w,
            temp=duration,
            guidance_scale=guidance_scale,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            cpu_offloading=False,
            save_memory=True,
            callback=lambda i, total: progress(i/total)
        )
    out_path = f"{uuid.uuid4()}_txt2vid.mp4"
    print(f"[INFO] Exporting video to {out_path}...")
    export_to_video(frames, out_path, fps=24)
    print("[INFO] Text-to-video complete.")
    return out_path


def generate_image_to_video(image, prompt, duration, video_guidance_scale, resolution, seed, progress=gr.Progress()):
    print("[INFO] generate_image_to_video called...")
    variant = '768p' if resolution == '768p' else '384p'
    h, w = (height_high, width_high) if variant == '768p' else (height_low, width_low)
    img = resize_crop_image(image, w, h)
    model, dtype = initialize_model_cached(variant, seed)
    print("[INFO] Starting image-to-video generation...")
    with torch.no_grad():
        frames = model.generate_i2v(
            prompt=prompt,
            input_image=img,
            num_inference_steps=[duration//2 + 1]*3,
            temp=duration,
            video_guidance_scale=video_guidance_scale,
            output_type="pil",
            cpu_offloading=False,
            save_memory=True,
            callback=lambda i, total: progress(i/total)
        )
    out_path = f"{uuid.uuid4()}_img2vid.mp4"
    print(f"[INFO] Exporting video to {out_path}...")
    export_to_video(frames, out_path, fps=24)
    print("[INFO] Image-to-video complete.")
    return out_path


def main():
    # Setup
    download_model(model_repo, download_dir, variants, required_file)
    preload_models()

    with gr.Blocks() as demo:
        gr.Markdown("# Pyramid Flow Video Generation (CPU Only)")
        res_dd = gr.Dropdown(choices=['384p','768p'], value='384p', label='Resolution')

        with gr.Tab("Text-to-Video"):
            txt_prompt = gr.Textbox(label="Prompt", lines=2)
            txt_dur = gr.Slider(1, 16, value=5, step=1, label="Duration")
            txt_gs = gr.Slider(1.0, 15.0, value=7.0, step=0.1, label="Guidance Scale")
            txt_vgs = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Video Guidance Scale")
            txt_seed = gr.Number(label="Seed (0=random)", value=0)
            txt_button = gr.Button("Generate Text2Video")
            txt_output = gr.Video(label="Output Video")
            txt_button.click(
                generate_text_to_video,
                [txt_prompt, txt_dur, txt_gs, txt_vgs, res_dd, txt_seed],
                txt_output
            )

        with gr.Tab("Image-to-Video"):
            img_input = gr.Image(type="pil", label="Input Image")
            img_prompt = gr.Textbox(label="Prompt", lines=2)
            img_dur = gr.Slider(1, 16, value=5, step=1, label="Duration")
            img_vgs = gr.Slider(1.0, 10.0, value=5.0, step=0.1, label="Video Guidance Scale")
            img_seed = gr.Number(label="Seed (0=random)", value=0)
            img_button = gr.Button("Generate Img2Video")
            img_output = gr.Video(label="Output Video")
            img_button.click(
                generate_image_to_video,
                [img_input, img_prompt, img_dur, img_vgs, res_dd, img_seed],
                img_output
            )

        demo.launch(share=False)

if __name__ == '__main__':
    main()
