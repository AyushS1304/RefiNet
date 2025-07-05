"""Enhanced Advanced Image Sharpening Teacher Model with NaN Fixes"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np
from PIL import Image
import glob
import random
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize console logging
def print(*args, **kwargs):
    return __builtins__.print(*args, **kwargs, flush=True)

# =================================
# 1. Advanced Model Architecture (Unchanged)
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
    """Multi-Scale Attention Fusion Network"""
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
# 2. Dataset with Enhanced Augmentation Safety
# =================================
class Vimeo90KDataset(Dataset):
    def __init__(self, data_path, patch_size=48, aug_intensity=1.0):
        self.data_path = data_path
        self.patch_size = patch_size
        self.aug_intensity = aug_intensity
        self.image_paths = self._load_image_paths()
        print(f"Loaded {len(self.image_paths)} images")
        
    def _load_image_paths(self):
        """Load image paths from the specified dataset path"""
        image_paths = []
        
        # Find all PNG images in the dataset
        search_path = os.path.join(self.data_path, "**", "*.png")
        image_paths = glob.glob(search_path, recursive=True)
        
        # If no images found, try alternative patterns
        if len(image_paths) == 0:
            print("No images found with recursive PNG search. Trying alternative patterns...")
            # Try looking in subdirectories
            for root, dirs, files in os.walk(self.data_path):
                for file in files:
                    if file.lower().endswith('.png'):
                        image_paths.append(os.path.join(root, file))
        
        print(f"Found {len(image_paths)} images in dataset")
        return image_paths
    
    def _augment(self, img):
        """Dynamic intensity augmentation with NaN protection"""
        img = np.array(img).astype(np.float32) / 255.0
        
        # Random flips
        if random.random() < 0.5*self.aug_intensity:
            img = np.flip(img, axis=1).copy()
        if random.random() < 0.5*self.aug_intensity:
            img = np.flip(img, axis=0).copy()
            
        # Random rotation
        if random.random() < 0.3*self.aug_intensity:
            angle = random.choice([90, 180, 270])
            img = np.rot90(img, k=angle//90).copy()
            
        # Color jitter with clamping
        if random.random() < 0.5*self.aug_intensity:
            brightness = 0.8 + 0.4*random.random()
            img = img * brightness
            img = np.clip(img, 0, 1)
            
        # Add noise with clamping
        if random.random() < 0.3*self.aug_intensity:
            noise = np.random.normal(0, 0.02*self.aug_intensity, img.shape)
            img = np.clip(img + noise, 0, 1)
            
        # Ensure no NaN/Inf values
        if np.isnan(img).any() or np.isinf(img).any():
            print("Warning: NaN/Inf detected after augmentation! Replacing with zeros.")
            img = np.zeros_like(img)
            
        return img
    
    def increase_augmentation(self):
        """Dynamically increase augmentation intensity with safety cap"""
        self.aug_intensity = min(1.5, self.aug_intensity + 0.1)
        print(f"Augmentation intensity increased to: {self.aug_intensity:.1f}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        try:
            hr_img = Image.open(self.image_paths[idx]).convert('RGB')
            hr_img = self._augment(hr_img)
            
            # Degradation simulation
            H, W, _ = hr_img.shape
            hr_pil = Image.fromarray((hr_img*255).astype(np.uint8))
            lr_pil = hr_pil.resize((max(1, W//4), max(1, H//4)), Image.BICUBIC)
            lr_pil = lr_pil.resize((W, H), Image.BICUBIC)
            lr_img = np.array(lr_pil) / 255.0
            
            # Extract patch
            x = random.randint(0, max(0, H - self.patch_size))
            y = random.randint(0, max(0, W - self.patch_size))
            hr_patch = hr_img[x:x+self.patch_size, y:y+self.patch_size]
            lr_patch = lr_img[x:x+self.patch_size, y:y+self.patch_size]
            
            # Final NaN check
            if np.isnan(hr_patch).any() or np.isnan(lr_patch).any():
                print(f"NaN detected in image {self.image_paths[idx]}! Using fallback.")
                hr_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
                lr_patch = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
            
            return (
                torch.tensor(lr_patch).permute(2, 0, 1).float(),
                torch.tensor(hr_patch).permute(2, 0, 1).float()
            )
        except Exception as e:
            print(f"Error loading image {self.image_paths[idx]}: {e}")
            # Return a random dummy image
            dummy = np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32)
            return (
                torch.tensor(dummy).permute(2, 0, 1).float(),
                torch.tensor(dummy).permute(2, 0, 1).float()
            )

# =================================
# 3. Enhanced Training Utilities
# =================================
class HybridLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.device = device
        
    def forward(self, pred, target):
        # Check for NaN inputs
        if torch.isnan(pred).any() or torch.isnan(target).any():
            print("Warning: NaN detected in loss calculation inputs!")
            return torch.tensor(0.1, device=self.device, requires_grad=True)
        
        # Pixel loss
        l1 = self.l1(pred, target)
        
        # Structural similarity with stabilization
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        EPSILON = 1e-8  # Stabilization term
        
        mu_x = pred.mean(dim=[1, 2, 3], keepdim=True)
        mu_y = target.mean(dim=[1, 2, 3], keepdim=True)
        
        sigma_x = pred.var(dim=[1, 2, 3], keepdim=True, unbiased=False) + EPSILON
        sigma_y = target.var(dim=[1, 2, 3], keepdim=True, unbiased=False) + EPSILON
        sigma_xy = ((pred - mu_x) * (target - mu_y)).mean(dim=[1, 2, 3], keepdim=True)
        
        numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + EPSILON
        
        ssim_map = numerator / denominator
        ssim_loss = 1 - ssim_map.mean()
        
        return 0.7 * l1 + 0.3 * ssim_loss

def centralize_gradients(model):
    """Gradient Centralization (GC)"""
    for param in model.parameters():
        if param.grad is not None:
            grad = param.grad.data
            if grad.dim() > 1:
                mean = torch.mean(grad, dim=tuple(range(1, grad.dim())), keepdim=True)
                param.grad.data = grad - mean

# =================================
# 4. Enhanced Plateau Detection & Handling
# =================================
class PlateauBreaker:
    def __init__(self, model, optimizer, dataloader):
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.best_ssim = 0
        self.plateau_count = 0
        self.max_lr = 3e-4  # Absolute maximum LR
        
    def __call__(self, current_ssim, epoch):
        if current_ssim > self.best_ssim + 0.001:
            self.best_ssim = current_ssim
            self.plateau_count = 0
            return False
            
        self.plateau_count += 1
        if self.plateau_count < 2:  # Allow 2 epochs plateau
            return False
            
        print(f"Plateau detected at epoch {epoch}! Applying countermeasures...")
        self.plateau_count = 0
        
        # 1. Conservative LR boost (capped)
        for g in self.optimizer.param_groups:
            new_lr = min(g['lr'] * 1.2, self.max_lr)  # 20% increase max
            g['lr'] = new_lr
        print(f"LR increased to {self.optimizer.param_groups[0]['lr']:.2e}")
        
        # 2. Increase augmentation
        self.dataloader.dataset.increase_augmentation()
        
        return True

# =================================
# 5. Training Loop with TQDM Bars
# =================================
def print_metrics(epoch, loss, val_ssim, lr, time_elapsed, gpu_mem, iterations):
    print(f"\n{'='*50}")
    print(f"Epoch {epoch} | Time: {time_elapsed:.1f}s")
    print(f"Loss: {loss:.4f} | SSIM: {val_ssim:.4f} | LR: {lr:.1e}")
    print(f"GPU Memory: {gpu_mem} | Iterations: {iterations}")
    print(f"{'='*50}\n")

# =================================
# 6. Enhanced Main Training Function
# =================================
def main():
    # Configuration
    DATA_PATH = "/kaggle/input/vimeo2/vimeo90(15)/sequences"
    TOTAL_EPOCHS = 42
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting training on {DEVICE}...")
    print(f"Dataset path: {DATA_PATH}")
    
    # Initialize model
    model = MSAFN().to(DEVICE)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
        print(f"Using {num_gpus} GPUs!")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params/1e6:.2f}M")
    
    # Loss and optimizer
    criterion = HybridLoss(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # Create dataset and loader
    try:
        full_dataset = Vimeo90KDataset(DATA_PATH, 48)
        full_loader = DataLoader(
            full_dataset, batch_size=BATCH_SIZE, 
            shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
        )
        print(f"Loaded {len(full_dataset)} images | Batches: {len(full_loader)}")
        
        # Check if dataset is empty
        if len(full_dataset) == 0:
            print("Error: No images found in dataset!")
            return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Scheduler
    total_steps = len(full_loader) * TOTAL_EPOCHS
    scheduler = OneCycleLR(
        optimizer, 
        max_lr=3e-4,
        total_steps=total_steps,
        pct_start=0.2,
        anneal_strategy='cos'
    )
    
    # Training state
    scaler = GradScaler()
    plateau_breaker = PlateauBreaker(model, optimizer, full_loader)
    best_ssim = 0
    iteration_counter = 0
    
    # GPU memory tracking
    def get_gpu_memory():
        if not torch.cuda.is_available():
            return "N/A"
        return f"{torch.cuda.memory_allocated()//1024**2}/{torch.cuda.max_memory_allocated()//1024**2} MB"
    
    # Start training
    for epoch in range(1, TOTAL_EPOCHS+1):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        skipped_batches = 0
        
        # Create progress bar for batches
        batch_bar = tqdm(
            total=len(full_loader), 
            desc=f"Epoch {epoch}/{TOTAL_EPOCHS}", 
            unit="batch",
            bar_format="{l_bar}{bar:30}{r_bar}",
            position=0
        )
        
        # Batch processing
        for batch_idx, (lr, hr) in enumerate(full_loader):
            iteration_counter += 1
            
            lr, hr = lr.to(DEVICE), hr.to(DEVICE)
            optimizer.zero_grad()
            
            # Skip batch if inputs contain NaN
            if torch.isnan(lr).any() or torch.isnan(hr).any():
                skipped_batches += 1
                batch_bar.set_postfix_str("Skipped NaN batch")
                batch_bar.update(1)
                continue
            
            with autocast():
                sr = model(lr)
                
                # Skip batch if output contains NaN
                if torch.isnan(sr).any():
                    skipped_batches += 1
                    batch_bar.set_postfix_str("Skipped NaN output")
                    batch_bar.update(1)
                    continue
                
                loss = criterion(sr, hr)
            
            # Skip batch if loss is NaN
            if torch.isnan(loss).any():
                skipped_batches += 1
                optimizer.zero_grad()
                batch_bar.set_postfix_str("Skipped NaN loss")
                batch_bar.update(1)
                continue
            
            scaler.scale(loss).backward()
            centralize_gradients(model)
            
            # Gradient clipping to prevent explosions
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            batch_bar.set_postfix(
                loss=f"{loss.item():.4f}", 
                lr=f"{optimizer.param_groups[0]['lr']:.1e}",
                gpu=get_gpu_memory(),
                skipped=skipped_batches
            )
            batch_bar.update(1)
        
        # Close batch bar
        batch_bar.close()
        
        # Validation
        model.eval()
        val_ssim = 0
        val_batches = min(20, len(full_loader))
        valid_val_batches = 0
        
        if val_batches == 0:
            print("No batches available for validation!")
            avg_val_ssim = 0
        else:
            # Create validation progress bar
            val_bar = tqdm(
                total=val_batches,
                desc="Validation",
                unit="batch",
                bar_format="{l_bar}{bar:30}{r_bar}",
                position=0,
                leave=True
            )
            
            with torch.no_grad():
                for i, (lr, hr) in enumerate(full_loader):
                    if i >= val_batches:
                        break
                        
                    lr, hr = lr.to(DEVICE), hr.to(DEVICE)
                    
                    # Skip validation batch with NaN
                    if torch.isnan(lr).any() or torch.isnan(hr).any():
                        val_bar.update(1)
                        continue
                        
                    sr = model(lr)
                    
                    # Skip validation batch with NaN output
                    if torch.isnan(sr).any():
                        val_bar.update(1)
                        continue
                    
                    # Manual SSIM calculation with stabilization
                    C1 = (0.01 * 1) ** 2
                    C2 = (0.03 * 1) ** 2
                    EPSILON = 1e-8
                    
                    mu_x = sr.mean(dim=[1, 2, 3], keepdim=True)
                    mu_y = hr.mean(dim=[1, 2, 3], keepdim=True)
                    
                    sigma_x = sr.var(dim=[1, 2, 3], keepdim=True, unbiased=False) + EPSILON
                    sigma_y = hr.var(dim=[1, 2, 3], keepdim=True, unbiased=False) + EPSILON
                    sigma_xy = ((sr - mu_x) * (hr - mu_y)).mean(dim=[1, 2, 3], keepdim=True)
                    
                    numerator = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
                    denominator = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2) + EPSILON
                    
                    ssim_map = numerator / denominator
                    val_ssim += ssim_map.mean().item()
                    valid_val_batches += 1
                    
                    # Update validation bar
                    val_bar.update(1)
            
            # Close validation bar
            val_bar.close()
            
            # Calculate metrics
            avg_val_ssim = val_ssim / valid_val_batches if valid_val_batches > 0 else 0
        
        # Calculate metrics
        actual_batches = len(full_loader) - skipped_batches
        avg_loss = total_loss / actual_batches if actual_batches > 0 else 0
        elapsed = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        gpu_mem = get_gpu_memory()
        
        # Print epoch summary
        print_metrics(epoch, avg_loss, avg_val_ssim, lr, elapsed, gpu_mem, iteration_counter)
        if skipped_batches > 0:
            print(f"Warning: Skipped {skipped_batches}/{len(full_loader)} batches due to NaN")
        
        # Plateau detection
        if plateau_breaker(avg_val_ssim, epoch):
            # Reset plateau counter after action
            pass
        
        # Save best model
        if avg_val_ssim > best_ssim:
            best_ssim = avg_val_ssim
            save_path = "/kaggle/working/best_model.pth"
            if num_gpus > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f"Saved best model with SSIM: {best_ssim:.4f}")
    
    # Final save
    save_path = "/kaggle/working/final_model.pth"
    if num_gpus > 1:
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
    
    print(f"Training complete! Best SSIM: {best_ssim:.4f}")
    
    # Create download link
    try:
        from IPython.display import FileLink
        print("\nDownload Model:", FileLink(save_path))
    except:
        print(f"Model saved at {save_path}")

if __name__ == "__main__":
    main()