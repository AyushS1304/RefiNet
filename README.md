# 📝 *Image Enhancement using Knowledge Distillation*

This project aims to **compress a powerful image enhancement model** (MSAFN) into a **lightweight student model** (LightMSAFN) using **knowledge distillation**. The goal is to retain most of the image quality (PSNR, SSIM) while reducing model size and computation time, making it ideal for **real-time enhancement** on resource-constrained devices.

- 📌 **Task**: Super-resolution / Image Enhancement  
- 🧠 **Teacher Model**: Multi-Scale Attention Fusion Network (MSAFN)  
- 🎓 **Student Model**: LightMSAFN (lightweight, fast, mobile-friendly)  
- 🔄 **Technique**: Knowledge distillation via combined pixel + soft loss  
- 📊 **Training Dataset**: Vimeo-90K (15 sequences subset with augmentation)  
- ✅ **Goal**: Achieve high SSIM (>0.94) and PSNR (~29 dB) in a compressed model

---

## 🧠 *Teacher Model (MSAFN)*
The **Multi-Scale Attention Fusion Network (MSAFN)** is an advanced teacher model designed for high-fidelity image sharpening and restoration. Built for knowledge distillation, it processes images through parallel **multi-scale pathways (48×48, 24×24, 12×12 resolutions)** with integrated channel attention gates that dynamically recalibrate feature importance. The architecture features stochastic depth residual blocks for robust feature extraction and a GRU-based recurrent refinement module that progressively enhances details through 3 iterative steps.

Engineered for stability during training, MSAFN includes **NaN-protected operations with automatic batch skipping**, dynamic augmentation scaling to combat performance plateaus, and gradient centralization for accelerated convergence. It employs a hybrid L1 + stabilized SSIM loss function and OneCycle LR scheduling (up to 3e-4) for optimal performance. The model processes Vimeo90K datasets efficiently in multi-GPU environments while maintaining VRAM usage under 12GB at 64 batch sizes, delivering state-of-the-art sharpening results ideal for distilling knowledge into lightweight student networks.

---

## 🧠 *Student Training with Knowledge Distillation (MSAFN → LightMSAFN)*

This PyTorch implementation presents a **lightweight Multi-Scale Attention Fusion Network (LightMSAFN)** trained via **knowledge distillation** from a powerful **MSAFN teacher model**, achieving efficient image enhancement/super-resolution on the **Vimeo-90K dataset**. The student model leverages logit distillation (KL divergence) and intermediate feature mimicking (MSE loss) to transfer knowledge while maintaining only 30% of the teacher's parameters through strategic architectural optimizations - **including channel reduction (64→32), shallower residual blocks (8→3), and elimination of recurrent components. The training protocol employs adaptive loss weighting (α=0.7 distillation + β=0.3 ground truth), OneCycle LR scheduling (2e-4 max), and mixed-precision acceleration, enabling the compact student to deliver comparable visual quality to the teacher at 2.8× faster inference speeds, making it ideal for edge deployment**. Critical enhancements like gradient centralization and progressive teacher guidance decay ensure stable convergence while preserving the teacher's restoration capabilities in a dramatically more efficient architecture.

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
- ~8.1M parameters

### 👨‍🎓 LightMSAFN (Student Model)
- Lightweight channel attention & residual blocks
- Efficient fusion and reduced-depth refinement
- Distilled from teacher using pixel + soft loss
- ~0.8M parameters

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

| Loss Type         | Description                                     |
|-------------------|-------------------------------------------------|
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
pip install streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
numpy==1.24.3
Pillow==10.0.0
opencv-python-headless==4.7.0.72
tqdm==4.65.0 

````

### 🧪 Run Training
```bash
!python3 teacher_training.py
``` 
---
```bash
!python3 student_training.py
```

### 🛠️ Configuration

| Parameter       | Value                    |
| --------------- | ------------------------ |
| Epochs          | 40                       |
| Batch Size      | 64                       |
| Patch Size      | 64X64                    |
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

| Metric    | Teacher (MSAFN) | Student (LightMSAFN) | Upon Validation |
| --------- | --------------- | -------------------- |-----------------|
| PSNR (dB) | \~29.6          | \~28.9               | \~51              |
| SSIM      | \~0.9423        | \~0.9416             | \~0.98           |
| Speed     | 1× (slow)       | ⚡ 3–4× faster        | ⚡4× faster    |
| Params    | \~8.1M          | \~0.8M               | \~0.03M         |

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
| **Ayush Sharma** |  Deep Learning Researcher(Teacher Model)              | [@AyushS1304](https://github.com/AyushS1304)           | [Ayush Sharma](http://www.linkedin.com/in/ayush-sharma-6219352b1)            |
| **Dhruv Agarwal** |  Deep Learning Researcher(Student Model)             | [@Dhruv610ag](https://github.com/Dhruv610ag)         | [Dhruv Agarwal](https://www.linkedin.com/in/dhruv-agarwal-773b32287/)           |
| **Aniket Shah**  | Frontend Developer(StreamLit)                       | [@Aniket200424](https://github.com/Aniket200424)           | [Aniket Shah](https://www.linkedin.com/in/aniket-shah-b1b008250/)        |

---

## 💬 Acknowledgements

* [Vimeo-90K Dataset](http://toflow.csail.mit.edu/)
* Inspired by works on lightweight SR and knowledge distillation in vision

---


You can now run this in a Kaggle or Colab notebook cell, and it will create a `README.md` file in your working directory. Let me know if you want to include diagrams, inference scripts, or visual results too.



```bash
git clone https://github.com/AyushS1304/RefiNet.git
```
---
