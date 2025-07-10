# ⚡ Pyramid Flow Video Generation (Fork)

> **An autoregressive video generation toolkit**  
> _Originally by [jy0205/Pyramid-Flow](https://github.com/jy0205/Pyramid-Flow), this repo is a private fork with Mac-specific CPU support._

---

## 📌 This Fork

- **Forked from**: [jy0205/Pyramid-Flow](https://github.com/jy0205/Pyramid-Flow)  
- **Custom script**:  
  - `app_cpu.py` is tailored for **macOS M1/M2** (Intel/Apple Silicon) and forces CPU/FP32 execution to avoid current MPS 3D‐Conv incompatibilities.

---

## 🚀 Features

- **Text-to-Video**  
- **Image-to-Video**  
- **Two resolutions**:  
  - 384 p (up to 5 s)  
  - 768 p (up to 10 s)  
- **Lightweight**: 20.7k A100-hours training  
- **Tiled VAE** for large-frame memory savings  

---

## 🛠️ Known MPS Limitations

Apple’s current MPS backend **does not support Conv3D** (used in our video VAE).  
- On M1/M2 you’ll either **offload the VAE to CPU** or **run fully on CPU** (`app_cpu.py`).  
- Attempts to run in half/bfloat16 on MPS will hit “Conv3D not supported” or “float64” errors.

---

## 📦 Installation

```bash
git clone https://github.com/YourUser/Pyramid-Flow.git
cd Pyramid-Flow

# set up Python venv
python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
