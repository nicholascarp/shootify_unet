#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:25:28 2025

@author: nicholascarp
"""

"""
Color degradation utilities for training data augmentation
"""

import torch


class ColorDegradation:
    """Fast color degradation for training"""
    
    @staticmethod
    def apply_random_degradation(image, mask, strength=0.5):
        """
        Apply purple/magenta color shift to masked region
        FAST - no checks, all GPU operations
        
        Args:
            image: torch.Tensor [3, H, W] in [0, 1]
            mask: torch.Tensor [1, H, W] in [0, 1]
            strength: float, degradation strength (default 0.5)
        
        Returns:
            degraded: torch.Tensor [3, H, W] in [0, 1]
        """
        # Color shift on same device as image (CRITICAL for speed)
        color_shift = torch.tensor(
            [0.35, -0.25, 0.35],  # R+, G-, B+ = purple/magenta
            device=image.device,
            dtype=image.dtype
        ).view(3, 1, 1)
        
        # Expand mask to 3 channels
        mask_3ch = mask.expand(3, -1, -1)
        
        # Apply degradation
        degraded = image + color_shift * mask_3ch * strength
        
        # Clamp to valid range
        return degraded.clamp(0, 1)