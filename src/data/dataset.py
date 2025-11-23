#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:24:17 2025

@author: nicholascarp
"""

"""
Dataset class for color correction training
"""

import csv
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from .degradation import ColorDegradation


class UpperMaskDegradedDataset(Dataset):
    """
    Dataset for color correction training
    Returns degraded (input) and clean (target) pairs
    """
    
    def __init__(self, manifest_csv, img_size=256, use_degradation=True, degradation_strength=0.5):
        """
        Args:
            manifest_csv: Path to CSV file with columns: image, mask_npy
            img_size: Target image size (default 256)
            use_degradation: Whether to apply color degradation (True for train, False for test)
            degradation_strength: Strength of color degradation (default 0.5)
        """
        self.items = []
        with open(manifest_csv, "r") as f:
            r = csv.DictReader(f)
            for row in r:
                self.items.append({
                    "image": Path(row["image"]),
                    "mask_npy": Path(row["mask_npy"])
                })
        
        self.img_size = img_size
        self.use_degradation = use_degradation
        self.degradation_strength = degradation_strength
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        it = self.items[idx]
        
        # Load image
        img_original = Image.open(it["image"]).convert("RGB")
        img_original = img_original.resize((self.img_size, self.img_size), Image.BILINEAR)
        img_original = TF.to_tensor(img_original)  # [3, H, W] in [0, 1]
        
        # Load mask
        mask = np.load(it["mask_npy"])
        if mask.shape[0] != self.img_size or mask.shape[1] != self.img_size:
            mask_pil = Image.fromarray(mask.astype(np.uint8)*255)
            mask_pil = mask_pil.resize((self.img_size, self.img_size), Image.NEAREST)
            mask = np.array(mask_pil) > 127
        
        mask = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)  # [1, H, W]
        
        # Apply degradation if enabled
        if self.use_degradation:
            img_degraded = ColorDegradation.apply_random_degradation(
                img_original,
                mask,
                strength=self.degradation_strength
            )
        else:
            img_degraded = img_original
        
        return {
            "image": img_degraded,      # INPUT: degraded colors (wrong)
            "gt": img_original,          # TARGET: correct colors
            "mask": mask,                # Garment mask
            "path": str(it["image"])
        }