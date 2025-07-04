import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from PIL import Image
import numpy as np

# Add parent directory to path to access utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from utils.teacher_training import MSAFN
from utils.student_training import LightMSAFN


class ImageEnhancer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.teacher = None
        self.student = None

    def load_models(self, teacher_path, student_path):
        """Load both teacher and student models"""
        teacher_path = os.path.abspath(teacher_path)
        student_path = os.path.abspath(student_path)

        self.teacher = self._load_model(teacher_path, 'teacher')
        print(f"✅ Loaded teacher model with {sum(p.numel() for p in self.teacher.parameters()):,} parameters")

        self.student = self._load_model(student_path, 'student')
        print(f"✅ Loaded student model with {sum(p.numel() for p in self.student.parameters()):,} parameters")

    def _load_model(self, model_path, model_type):
        """Load pre-trained model with architecture detection"""
        try:
            model = MSAFN() if model_type == 'teacher' else LightMSAFN()
            model = model.to(self.device)

            state_dict = torch.load(model_path, map_location=self.device)

            # Remove "module." prefix if trained with DataParallel
            if all(k.startswith("module.") for k in state_dict):
                state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            model.load_state_dict(state_dict)
            model.eval()
            return model

        except Exception as e:
            raise RuntimeError(f"❌ Error loading {model_type} model: {str(e)}")

    def enhance_image(self, lr_img, model_type='student', scale_factor=4):
        """Enhance image using specified model"""

        # Convert to RGB if RGBA or other mode
        if isinstance(lr_img, str):
            lr_img = Image.open(lr_img).convert('RGB')
        elif isinstance(lr_img, Image.Image):
            lr_img = lr_img.convert('RGB')

        # Select model
        model = self.teacher if model_type == 'teacher' else self.student

        if model is None:
            raise ValueError("Model not loaded! Call load_models() first.")

        # Preprocessing
        original_size = lr_img.size
        lr_np = np.array(lr_img).astype(np.float32) / 255.0  # [H, W, C]
        lr_tensor = torch.from_numpy(lr_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        # Sanity check
        assert lr_tensor.shape[1] == 3, f"Expected 3 channels, got {lr_tensor.shape[1]}"

        # Inference
        with torch.no_grad():
            sr_tensor = model(lr_tensor).clamp(0, 1)

        # Postprocessing
        sr_np = sr_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255
        sr_img = Image.fromarray(sr_np.astype(np.uint8))

        # Resize if scale != 4
        if scale_factor != 4:
            target_size = (int(original_size[0] * scale_factor),
                           int(original_size[1] * scale_factor))
            sr_img = sr_img.resize(target_size, Image.BICUBIC)

        return sr_img
