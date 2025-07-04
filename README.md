# 📝 Project Info — *Image Enhancement using Knowledge Distillation*

This project aims to **compress a powerful image enhancement model** (MSAFN) into a **lightweight student model** (LightMSAFN) using **knowledge distillation**. The goal is to retain most of the image quality (PSNR, SSIM) while reducing model size and computation time, making it ideal for **real-time enhancement** on resource-constrained devices.

- 📌 **Task**: Super-resolution / Image Enhancement  
- 🧠 **Teacher Model**: Multi-Scale Attention Fusion Network (MSAFN)  
- 🎓 **Student Model**: LightMSAFN (lightweight, fast, mobile-friendly)  
- 🔄 **Technique**: Knowledge distillation via combined pixel + soft loss  
- 📊 **Training Dataset**: Vimeo-90K (15 sequences subset with augmentation)  
- ✅ **Goal**: Achieve high SSIM (>0.92) and PSNR (~31 dB) in a compressed model

---

# 🧠 Student Training with Knowledge Distillation (MSAFN → LightMSAFN)

A PyTorch-based implementation of a lightweight **Multi-Scale Attention Fusion Network (LightMSAFN)** trained using **knowledge distillation** from a deeper **MSAFN teacher model** for efficient **image enhancement / super-resolution** on the **Vimeo-90K** dataset.

---

## ✨ Highlights

- 🔥 **Teacher**: MSAFN — deep, multi-scale residual transformer-like model  
- ⚡ **Student**: LightMSAFN — compressed and fast model with comparable performance  
- 🎓 **Knowledge Distillation**: Balanced L1 + Soft Loss from teacher predictions  
- 🧪 **Mixed Precision Training**: Faster training with AMP (`autocast`)  
- 📊 **Evaluation**: SSIM + PSNR tracking with best model checkpointing  
- 🧼 **NaN-safe Augmentations**: Resilient training with strong image augmentations  
- 🏗️ **Modular Design**: Easily extensible and clean training pipeline

---

## 🏗️ Architecture Overview

### 👨‍🏫 MSAFN (Teacher Model)
- Multi-scale processing (1×, 2×, 4× downsampling)
- Residual Dense Blocks with attention gates
- Recurrent refinement via GRU-like module
- ~2.3M parameters

### 👨‍🎓 LightMSAFN (Student Model)
- Lightweight channel attention & residual blocks
- Efficient fusion and reduced-depth refinement
- Distilled from teacher using pixel + soft loss
- ~0.3M parameters

---

## 🗂️ Dataset

- 📁 **Vimeo-90K** (custom subset)
- Resolution: 256×256 crops
- Format: Raw `.png` sequences
- Data Augmentations:
  - Random flips & rotations
  - Brightness jitter
  - Gaussian noise
  - Bicubic downsample + upscale for LR generation

## 📦 Vimeo-90K Dataset
- The Vimeo-90K dataset is a large-scale, high-quality video dataset commonly used for video enhancement tasks such as video super-resolution, frame interpolation, and video denoising. It was introduced in the paper:

- TOFlow: Video Enhancement with Task-Oriented Flow
Tianfan Xue, Baian Chen, Jiajun Wu, Donglai Wei, William T. Freeman

## 📁 Structure
- The dataset contains 91,701 video clips, each consisting of 7 consecutive frames (448×256 resolution). It includes two main subsets:

- Vimeo-90K Septuplet – used for tasks like super-resolution, denoising, and deblurring

- Vimeo-90K Triplet – often used for video frame interpolation

- Each clip is organized in a folder containing PNG images named im1.png through im7.png.


## 🔍 Applications
- Video Super-Resolution

- Frame Interpolation

- Motion Compensation

- Video Denoising

- Optical Flow Estimation



---

## 🧪 Loss Functions

| Loss Type         | Description                                      |
|------------------|--------------------------------------------------|
| `L1`              | Student vs Ground Truth (pixel reconstruction)  |
| `MSE`             | Student vs Teacher Output (soft guidance)       |
| `DistillationLoss`| Combined: `alpha * L1 + (1-alpha) * MSE`        |

`alpha = 0.5` (can be tuned)

---

## 📈 Metrics

- ✅ **PSNR** (Peak Signal-to-Noise Ratio)
- ✅ **SSIM** (Structural Similarity Index)
- ✅ Logged per epoch + visualized via tqdm bar

---

## 🚀 Training Pipeline

### 📦 Requirements
```bash
pip install torch torchvision numpy pillow tqdm
````

### 🧪 Run Training

```bash
python student_training.py
```

### 🛠️ Configuration

| Parameter       | Value                    |
| --------------- | ------------------------ |
| Epochs          | 35                       |
| Batch Size      | 64                       |
| Patch Size      | 48×48                    |
| Optimizer       | AdamW                    |
| Scheduler       | ReduceLROnPlateau        |
| Mixed Precision | ✅ Yes (`torch.cuda.amp`) |
| LR              | 1e-3 (with decay)        |
| Gradient Clip   | 0.5                      |
| GPUs Used       | Auto (`nn.DataParallel`) |

---

## 🧠 File Structure

```
student_training.py
├── MSAFN           # Teacher model
├── LightMSAFN      # Student model
├── Vimeo90KDataset # Dataset with strong augmentation
├── DistillationLoss# Custom loss combining L1 & MSE
├── Training Loop   # AMP, distillation, metric tracking
└── Model Saving    # Best & final model checkpoints
```

---

## 🏁 Sample Results

| Metric    | Teacher (MSAFN) | Student (LightMSAFN) |
| --------- | --------------- | -------------------- |
| PSNR (dB) | \~31.5          | \~30.8               |
| SSIM      | \~0.955         | \~0.928              |
| Speed     | 1× (slow)       | ⚡ 3–4× faster        |
| Params    | \~2.3M          | \~0.3M               |

---

## 📦 Model Checkpoints

| Path                                      | Description        |
| ----------------------------------------- | ------------------ |
| `/kaggle/input/msafncustom/*.pth`         | Pretrained teacher |
| `/kaggle/working/best_student_*.pth`      | Best student model |
| `/kaggle/working/final_student_model.pth` | Final checkpoint   |

---

## ✍️ Authors

| Name            | Role                                | GitHub                                                | LinkedIn                                                |
|-----------------|-------------------------------------|--------------------------------------------------------|----------------------------------------------------------|
| **Ayush Sharma** | Java & Deep Learning Researcher     | [@AyushS1304](https://github.com/AyushS1304)           | [Ayush Sharma](https://linkedin.com/in/Ayush)            |
| **Dhruv Agarwal** | Image Enhancement & DL Researcher  | [@Dhruv610ag](https://github.com/yourgithub)           | [Dhruv Agarwal](https://linkedin.com/in/dhruv)           |
| **Aniket Shah**  | Frontend Developer                  | [@Aniket200424](https://github.com/Aniket200424)           | [Aniket Shah](https://linkedin.com/in/AniketShah)        |

---

## 💬 Acknowledgements

* [Vimeo-90K Dataset](http://toflow.csail.mit.edu/)
* Inspired by works on lightweight SR and knowledge distillation in vision

---

> “Knowledge distilled is power amplified.” – Your Student Model 😄

```

---

You can now run this in a Kaggle or Colab notebook cell, and it will create a `README.md` file in your working directory. Let me know if you want to include diagrams, inference scripts, or visual results too.
```