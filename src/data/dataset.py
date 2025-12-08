"""
Dataset for color correction with cloth reference
V2.2: Lighting-aware cloth (combines color + lighting pattern)
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from .degradation import ColorDegradation


class UpperMaskDegradedDataset(Dataset):
   """
    Production-quality dataset for color transfer
    
    Key features:
    1. Lighting-aware cloth (mean color + luminance pattern)
    2. Random color degradation with guaranteed visibility
    3. Cloth mismatching augmentation (30% probability)
    """
    
    def __init__(
        self,
        manifest_csv: str,
        img_size: int = 256,
        use_degradation: bool = True,
        degradation_strength: float = 0.5,
        cloth_mismatch_prob: float = 0.3,
    ):
        """
        Args:
            manifest_csv: CSV with columns: image, mask_path, cloth_image
            img_size: Target image size
            use_degradation: Whether to apply color degradation
            degradation_strength: Strength of degradation (0.0-1.0)
            cloth_mismatch_prob: Probability of using mismatched cloth
        """
        super().__init__()
        
        self.img_size = img_size
        self.use_degradation = use_degradation
        self.degradation_strength = degradation_strength
        self.cloth_mismatch_prob = cloth_mismatch_prob
        
        # Read manifest CSV
        if not os.path.exists(manifest_csv):
            raise FileNotFoundError(f"Manifest not found: {manifest_csv}")
        
        df = pd.read_csv(manifest_csv)
        
        # Accept either 'mask_npy' or 'mask_path' column name
        mask_col = 'mask_npy' if 'mask_npy' in df.columns else 'mask_path'
        
        # Validate required columns
        required_cols = ['image', mask_col, 'cloth_image']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Manifest missing columns: {missing}")
        
        self.image_paths = df['image'].tolist()
        self.mask_paths = df[mask_col].tolist()
        self.cloth_paths = df['cloth_image'].tolist()
        
        print(f"Loaded {len(self.image_paths)} samples from {manifest_csv}")
        
        # Detect mask format
        first_mask = self.mask_paths[0]
        self.mask_format = self._detect_mask_format(first_mask)
        print(f"Mask format: {self.mask_format}")
        
        if use_degradation:
            print(f"Degradation: ENABLED (strength={degradation_strength})")
            print(f"Cloth mismatch: {cloth_mismatch_prob*100:.0f}%")
        else:
            print(f"Degradation: DISABLED")
    
    def _detect_mask_format(self, path: str) -> str:
        """Detect mask format from file extension"""
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            return 'npy'
        elif ext in ['.png', '.jpg', '.jpeg']:
            return 'image'
        else:
            raise ValueError(f"Unsupported mask format: {ext}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def _load_image(self, path: str) -> torch.Tensor:
        """Load and preprocess image"""
        img = Image.open(path).convert('RGB')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        img = TF.to_tensor(img)  # [3, H, W] in [0, 1]
        return img
    
    def _load_mask(self, path: str) -> torch.Tensor:
        """Load segmentation mask and extract upper clothes (Class 5)"""
        if self.mask_format == 'npy':
            mask = np.load(path)
            while mask.ndim > 2:
                mask = mask[0]
        else:
            # Load palette image
            mask_pil = Image.open(path)
            
            if mask_pil.mode == 'P':
                mask = np.array(mask_pil, dtype=np.uint8)
            else:
                mask = np.array(mask_pil.convert('L'), dtype=np.uint8)
            
            # Resize
            if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
                mask_pil_resized = Image.fromarray(mask)
                mask_pil_resized = mask_pil_resized.resize(
                    (self.img_size, self.img_size), 
                    Image.NEAREST
                )
                mask = np.array(mask_pil_resized, dtype=np.uint8)
        
        # Extract class 5 (upper clothes)
        mask = (mask == 5).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]
        
        return mask
    
    def _create_lighting_aware_cloth(self, cloth: torch.Tensor, on_model: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Create lighting-aware cloth reference
        
        Combines:
        - Color on Cloth image + Lighting pattern from on-model garment 
        
        Args:
            cloth: [3, H, W] - cloth reference image
            on_model: [3, H, W] - on-model image
            mask: [1, H, W] - garment mask
        
        Returns:
            cloth_with_lighting: [3, H, W] - cloth color with lighting pattern
        """
        # 1. Extract mean color from cloth (target color)
        cloth_mean_color = cloth.mean(dim=[1, 2])  # [3] - RGB mean
        
        # 2. Extract luminance from on-model garment (lighting pattern)
        # Standard luminance conversion: 0.299*R + 0.587*G + 0.114*B
        on_model_luminance = (
            0.299 * on_model[0] + 
            0.587 * on_model[1] + 
            0.114 * on_model[2]
        )  # [H, W]
        
        # 3. Normalize luminance to reasonable range
        # Get luminance statistics from masked region only
        masked_lum = on_model_luminance * mask.squeeze(0)
        
        if mask.sum() > 0:
            # Get min/max from masked region
            mask_bool = mask.squeeze(0) > 0.5
            lum_values = on_model_luminance[mask_bool]
            
            lum_min = lum_values.min()
            lum_max = lum_values.max()
            
            # Normalize to 0.3-1.3 range (preserve relative brightness)
            # This keeps shadows dark and highlights bright
            if lum_max > lum_min:
                normalized_lum = (on_model_luminance - lum_min) / (lum_max - lum_min + 1e-6)
                normalized_lum = normalized_lum * 1.0 + 0.3  # Range: [0.3, 1.3]
            else:
                normalized_lum = torch.ones_like(on_model_luminance)
        else:
            normalized_lum = torch.ones_like(on_model_luminance)
        
        # 4. Apply cloth color modulated by lighting pattern
        cloth_with_lighting = cloth_mean_color.view(3, 1, 1) * normalized_lum.unsqueeze(0)
        
        # Clamp to valid range
        cloth_with_lighting = cloth_with_lighting.clamp(0, 1)
        
        
        # Cloth mean color: Red [0.8, 0.1, 0.1]
        # Normalized lum: Varies 0.3 (shadow) to 1.3 (highlight)
        
        # Shadow pixel (lum=0.3):
        #   R: 0.8 × 0.3 = 0.24 (dark red)
        #   G: 0.1 × 0.3 = 0.03
        #   B: 0.1 × 0.3 = 0.03
        
        # Highlight pixel (lum=1.0):
        #   R: 0.8 × 1.0 = 0.8 (bright red)
        #   G: 0.1 × 1.0 = 0.1
        #   B: 0.1 × 1.0 = 0.1
        
        return cloth_with_lighting


    def __getitem__(self, idx):
        """
        Returns training sample with lighting-aware cloth reference
        """
        # Load on-model image and mask
        on_model_path = self.image_paths[idx]
        on_model = self._load_image(on_model_path)
        
        mask_path = self.mask_paths[idx]
        mask = self._load_mask(mask_path)
        
        # Choose cloth (with mismatch augmentation)
        if self.use_degradation and np.random.random() < self.cloth_mismatch_prob:
            # Use different cloth
            random_idx = np.random.randint(0, len(self.cloth_paths))
            cloth_path = self.cloth_paths[random_idx]
        else:
            # Use matching cloth
            cloth_path = self.cloth_paths[idx]
        
        # Load cloth
        cloth = self._load_image(cloth_path)
        
        # Create lighting-aware cloth reference
        # This combines cloth COLOR with on-model LIGHTING
        cloth_with_lighting = self._create_lighting_aware_cloth(cloth, on_model, mask)
        
        # Ground truth
        gt = on_model.clone()
        
        # Apply degradation
        if self.use_degradation:
            degraded = ColorDegradation.apply_realistic_degradation(
                on_model, 
                mask, 
                strength=self.degradation_strength
            )
        else:
            degraded = on_model
        
        return {
            'image': degraded,               # Degraded on-model
            'mask': mask,                    # Upper clothes mask
            'gt': gt,                        # Original on-model
            'cloth': cloth_with_lighting     # Lighting-aware cloth!
        }