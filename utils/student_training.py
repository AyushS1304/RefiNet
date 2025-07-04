import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import random
import warnings
import math

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize console logging
print = lambda *args, **kwargs: __builtins__.print(*args, **kwargs, flush=True)

# =================================
# 1. Teacher Model Architecture (for loading)
# =================================
class MultiScaleGate(nn.Module):
    """Multi-scale channel attention gate"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ResidualDenseBlock(nn.Module):
    """Stochastic depth residual dense block"""
    def __init__(self, channels, drop_prob=0.2):
        super().__init__()
        self.drop_prob = drop_prob
        
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels*3, channels, 3, padding=1)
        self.gate = MultiScaleGate(channels)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        if self.training and random.random() < self.drop_prob:
            return x
            
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        return x + self.gate(x3)

class RecurrentRefinement(nn.Module):
    """GRU-based refinement module"""
    def __init__(self, channels):
        super().__init__()
        self.conv_z = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv_r = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv_h = nn.Conv2d(channels*2, channels, 3, padding=1)
        
    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
            
        xh = torch.cat([x, h], 1)
        z = torch.sigmoid(self.conv_z(xh))
        r = torch.sigmoid(self.conv_r(xh))
        h_hat = torch.tanh(self.conv_h(torch.cat([x, r*h], 1)))
        return (1 - z) * h + z * h_hat

class MSAFN(nn.Module):
    """Multi-Scale Attention Fusion Network (Teacher)"""
    def __init__(self):
        super().__init__()
        # Initial feature extraction
        self.conv_init = nn.Conv2d(3, 64, 3, padding=1)
        
        # Multi-scale processing paths
        self.scale1 = nn.Sequential(
            ResidualDenseBlock(64),
            MultiScaleGate(64)
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(2),
            ResidualDenseBlock(64),
            MultiScaleGate(64)
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(4),
            ResidualDenseBlock(64),
            MultiScaleGate(64)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Residual dense blocks
        self.res_blocks = nn.Sequential(
            *[ResidualDenseBlock(128) for _ in range(8)]
        )
        
        # Recurrent refinement
        self.refinement = RecurrentRefinement(128)
        
        # Reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 3, padding=1)
        )
    
    def forward(self, x):
        # Initial features
        x0 = self.conv_init(x)
        
        # Multi-scale processing
        s1 = self.scale1(x0)
        s2 = self.scale2(x0)
        s3 = self.scale3(x0)
        
        # Upsample and align scales
        s2 = F.interpolate(s2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=s1.shape[2:], mode='bilinear', align_corners=False)
        
        # Feature fusion
        features = self.fusion(torch.cat([s1, s2, s3], dim=1))
        
        # Residual processing
        features = self.res_blocks(features)
        
        # Recurrent refinement (3 steps)
        h = None
        for _ in range(3):
            h = self.refinement(features, h)
        features = h
        
        # Reconstruction
        return self.recon(features)

# =================================
# 2. Lightweight Student Model Architecture
# =================================
class LightweightGate(nn.Module):
    """Lightweight channel attention gate"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction, channels),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class LightResidualBlock(nn.Module):
    """Lightweight residual block"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gate = LightweightGate(channels)
        self.lrelu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        identity = x
        x = self.lrelu(self.conv1(x))
        x = self.conv2(x)
        return identity + self.gate(x)

class LightRecurrent(nn.Module):
    """Light GRU-based refinement"""
    def __init__(self, channels):
        super().__init__()
        self.conv_z = nn.Conv2d(channels*2, channels, 3, padding=1)
        self.conv_h = nn.Conv2d(channels*2, channels, 3, padding=1)
        
    def forward(self, x, h):
        if h is None:
            h = torch.zeros_like(x)
            
        xh = torch.cat([x, h], 1)
        z = torch.sigmoid(self.conv_z(xh))
        h_hat = torch.tanh(self.conv_h(torch.cat([x, h], 1)))
        return (1 - z) * h + z * h_hat

class LightMSAFN(nn.Module):
    """Lightweight Multi-Scale Attention Fusion Network"""
    def __init__(self):
        super().__init__()
        # Initial feature extraction
        self.conv_init = nn.Conv2d(3, 32, 3, padding=1)
        
        # Multi-scale processing paths
        self.scale1 = nn.Sequential(
            LightResidualBlock(32),
            LightweightGate(32)
        )
        self.scale2 = nn.Sequential(
            nn.AvgPool2d(2),
            LightResidualBlock(32),
            LightweightGate(32)
        )
        self.scale3 = nn.Sequential(
            nn.AvgPool2d(4),
            LightResidualBlock(32),
            LightweightGate(32)
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(96, 64, 1),
            nn.LeakyReLU(0.2)
        )
        
        # Residual blocks
        self.res_blocks = nn.Sequential(
            *[LightResidualBlock(64) for _ in range(3)]
        )
        
        # Recurrent refinement
        self.refinement = LightRecurrent(64)
        
        # Reconstruction
        self.recon = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 3, 3, padding=1)
        )
    
    def forward(self, x):
        # Initial features
        x0 = self.conv_init(x)
        
        # Multi-scale processing
        s1 = self.scale1(x0)
        s2 = self.scale2(x0)
        s3 = self.scale3(x0)
        
        # Upsample and align scales
        s2 = F.interpolate(s2, size=s1.shape[2:], mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=s1.shape[2:], mode='bilinear', align_corners=False)
        
        # Feature fusion
        features = self.fusion(torch.cat([s1, s2, s3], dim=1))
        
        # Residual processing
        features = self.res_blocks(features)
        
        # Recurrent refinement (2 steps)
        h = None
        for _ in range(2):
            h = self.refinement(features, h)
        features = h
        
        # Reconstruction
        return self.recon(features)

# =================================
# 3. Dataset
# =================================
class Vimeo90KDataset(Dataset):
    def __init__(self, data_path, patch_size=48, aug_intensity=1.0):
        self.data_path = data_path
        self.patch_size = patch_size
        self.aug_intensity = aug_intensity
        self.image_paths = self._load_image_paths()
        print(f"Loaded {len(self.image_paths)} images")
        
    def _load_image_paths(self):
        image_paths = []
        search_path = os.path.join(self.data_path, "**", "*.png")
        image_paths = glob.glob(search_path, recursive=True)
        
        if len(image_paths) == 0:
            print("No images found with recursive PNG search. Trying alternative patterns...")
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.lower().endswith('.png'):
                        image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images in dataset")
        return image_paths
    
    def _augment(self, img):
        img = np.array(img).astype(np.float32) / 255.0
        
        if random.random() < 0.5*self.aug_intensity:
            img = np.flip(img, axis=1).copy()
        if random.random() < 0.5*self.aug_intensity:
            img = np.flip(img, axis=0).copy()
            
        if random.random() < 0.3*self.aug_intensity:
            angle = random.choice([90, 180, 270])
            img = np.rot90(img, k=angle//90).copy()
            
        if random.random() < 0.5*self.aug_intensity:
            brightness = 0.8 + 0.4*random.random()
            img = img * brightness
            img = np.clip(img, 0, 1)
            
        if random.random() < 0.3*self.aug_intensity:
            noise = np.random.normal(0, 0.02*self.aug_intensity, img.shape)
            img = np.clip(img + noise, 0, 1)
            
        if np.isnan(img).any() or np.isinf(img).any():
            img = np.zeros_like(img)
            
        return img
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            hr_img = Image.open(self.image_paths[idx]).convert('RGB')
            hr_img = self._augment(hr_img)
            
            H, W, _ = hr_img.shape
            hr_pil = Image.fromarray((hr_img*255).astype(np.uint8))
            lr_pil = hr_pil.resize((max(1, W//4), max(1, H//4)), Image.BICUBIC)
            lr_pil = lr_pil.resize((W, H), Image.BICUBIC)
            lr_img = np.array(lr_pil) / 255.0
            
            x = random.randint(0, max(0, H - self.patch_size))
            y = random.randint(0, max(0, W - self.patch_size))
            hr_patch = hr_img[x:x+self.patch_size, y:y+self.patch_size]
            lr_patch = lr_img[x:x+self.patch_size, y:y+self.patch_size]
            
            if np.isnan(hr_patch).any() or np.isnan(lr_patch).any():
                hr_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
                lr_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
            
            return (
                torch.tensor(lr_patch).permute(2, 0, 1).float(),
                torch.tensor(hr_patch).permute(2, 0, 1).float()
            )
        except Exception as e:
            dummy = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
            return (
                torch.tensor(dummy).permute(2, 0, 1).float(),
                torch.tensor(dummy).permute(2, 0, 1).float()
            )

# =================================
# 4. Knowledge Distillation Loss
# =================================
class DistillationLoss(nn.Module):
    def __init__(self, device, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.device = device
        
    def forward(self, student_out, teacher_out, target):
        # Pixel loss (student vs target)
        pixel_loss = self.l1(student_out, target)
        
        # Distillation loss (soft targets)
        soft_loss = self.mse(student_out, teacher_out.detach())
        
        # Combined loss
        total_loss = self.alpha * pixel_loss + (1 - self.alpha) * soft_loss
        return total_loss, pixel_loss, soft_loss

# =================================
# 5. Training Utilities
# =================================
def centralize_gradients(model):
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.data
            if grad.dim() > 1:
                mean = torch.mean(grad, dim=tuple(range(1, grad.dim())), keepdim=True)
                param.grad.data = grad - mean

def get_gpu_memory():
    if not torch.cuda.is_available():
        return "N/A"
    return f"{torch.cuda.memory_allocated()//1024**2}/{torch.cuda.max_memory_allocated()//1024**2} MB"

def calculate_psnr(img1, img2):
    # Correct PSNR calculation
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def calculate_ssim(img1, img2):
    # Correct SSIM calculation
    C1 = (0.01 * 1) ** 2
    C2 = (0.03 * 1) ** 2
    mu1 = torch.mean(img1, dim=[1, 2, 3], keepdim=True)
    mu2 = torch.mean(img2, dim=[1, 2, 3], keepdim=True)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = torch.var(img1, dim=[1, 2, 3], keepdim=True, unbiased=False)
    sigma2_sq = torch.var(img2, dim=[1, 2, 3], keepdim=True, unbiased=False)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2), dim=[1, 2, 3], keepdim=True)
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

# =================================
# 6. Training Loop with Distillation
# =================================
def main():
    # Configuration
    DATA_PATH = "/kaggle/input/vimeo6/vimeo90(15)/sequences"
    TEACHER_PATH = "/kaggle/input/msafncustom/best_model (1).pth"
    TOTAL_EPOCHS = 35
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting training on {DEVICE}...")
    print(f"Dataset path: {DATA_PATH}")
    
    # Initialize teacher model
    teacher = MSAFN().to(DEVICE)
    teacher_state_dict = torch.load(TEACHER_PATH, map_location=DEVICE)
    
    # Handle DataParallel if used in teacher training
    if all(k.startswith('module.') for k in teacher_state_dict):
        teacher_state_dict = {k.replace('module.', ''): v for k, v in teacher_state_dict.items()}
    
    teacher.load_state_dict(teacher_state_dict)
    teacher.eval()
    print("Loaded pre-trained teacher model")
    
    # Initialize student model
    student = LightMSAFN().to(DEVICE)
    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)
        print(f"Using {torch.cuda.device_count()} GPUs for student!")
    
    # Print model sizes
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"Teacher parameters: {teacher_params/1e6:.2f}M")
    print(f"Student parameters: {student_params/1e6:.2f}M")
    
    # Loss and optimizer
    criterion = DistillationLoss(DEVICE, alpha=0.5)  # Balanced distillation weight
    optimizer = optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # Dataset and loader
    dataset = Vimeo90KDataset(DATA_PATH, 48)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, 
        shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Training state
    scaler = GradScaler()
    best_ssim = 0
    best_psnr = 0
    
    # Create model directory
    os.makedirs("/kaggle/working/models", exist_ok=True)
    
    # Start training
    for epoch in range(1, TOTAL_EPOCHS+1):
        epoch_start = time.time()
        student.train()
        total_loss = 0
        total_pixel_loss = 0
        total_soft_loss = 0
        total_ssim = 0
        total_psnr = 0
        batches = 0
        
        # Training loop
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{TOTAL_EPOCHS}")
        for lr, hr in pbar:
            lr, hr = lr.to(DEVICE, non_blocking=True), hr.to(DEVICE, non_blocking=True)
            
            # Teacher prediction
            with torch.no_grad():
                teacher_out = teacher(lr)
            
            # Student prediction
            optimizer.zero_grad()
            with autocast():
                student_out = student(lr)
                loss, pixel_loss, soft_loss = criterion(student_out, teacher_out, hr)
            
            # Backpropagation
            scaler.scale(loss).backward()
            centralize_gradients(student)
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()
            
            # Metrics
            total_loss += loss.item()
            total_pixel_loss += pixel_loss.item()
            total_soft_loss += soft_loss.item()
            
            # Calculate SSIM and PSNR
            with torch.no_grad():
                ssim_val = calculate_ssim(student_out, hr)
                psnr_val = calculate_psnr(student_out, hr)
                total_ssim += ssim_val.item()
                total_psnr += psnr_val.item()
            
            batches += 1
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                ssim=f"{ssim_val.item():.4f}", 
                psnr=f"{psnr_val.item():.2f}",
                pix_loss=f"{pixel_loss.item():.4f}",
                soft_loss=f"{soft_loss.item():.4f}"
            )
        
        # Epoch metrics
        avg_loss = total_loss / batches
        avg_pixel_loss = total_pixel_loss / batches
        avg_soft_loss = total_soft_loss / batches
        avg_ssim = total_ssim / batches
        avg_psnr = total_psnr / batches
        epoch_time = time.time() - epoch_start
        
        # Update scheduler
        scheduler.step(avg_ssim)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"Total Loss: {avg_loss:.4f} | Pixel Loss: {avg_pixel_loss:.4f} | Distillation Loss: {avg_soft_loss:.4f}")
        print(f"SSIM: {avg_ssim:.4f} | PSNR: {avg_psnr:.2f} dB")
        print(f"Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s | LR: {optimizer.param_groups[0]['lr']:.1e}")
        
        # Save model after every epoch
        #epoch_save_path = f"/kaggle/working/models/student_epoch_{epoch}_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
        #torch.save(student.state_dict(), epoch_save_path)
        #print(f"Saved model for epoch {epoch}")
        
        # Update best model if both metrics improve
        if avg_ssim > best_ssim and avg_psnr > best_psnr:
            best_ssim = avg_ssim
            best_psnr = avg_psnr
            best_save_path = f"/kaggle/working/best_student_ssim{avg_ssim:.4f}_psnr{avg_psnr:.2f}.pth"
            torch.save(student.state_dict(), best_save_path)
            print(f"Saved new best model with SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f}")
    
    # Save final model
    final_save_path = "/kaggle/working/final_student_model.pth"
    torch.save(student.state_dict(), final_save_path)
    print(f"Saved final student model")
    
    print(f"\nTraining complete!")
    print(f"Best SSIM: {best_ssim:.4f} | Best PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    main()

