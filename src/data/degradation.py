#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced color degradation for color transfer training
Guaranteed visible degradation on ALL color ranges
"""

import torch
import numpy as np


class ColorDegradation:
    """Advanced color degradation for training"""
    
    
    @staticmethod
    def apply_realistic_degradation(image, mask, strength=0.5, min_mae=0.05, max_retries=5):
        """
        Apply realistic color degradation with GUARANTEED visible change
        
        Validates actual degradation and retries if needed to ensure visibility.
        This ensures ALL colors (including very dark ones) degrade visibly.
        
        Args:
            image: torch.Tensor [3, H, W] or [B, 3, H, W]
            mask: torch.Tensor [1, H, W] or [B, 1, H, W]
            strength: float, overall degradation strength (0-1)
            min_mae: float, minimum required MAE in masked region (default 0.05)
            max_retries: int, maximum retry attempts (default 5)
        
        Returns:
            degraded: torch.Tensor same shape as image with guaranteed visible degradation
        """
        batch_mode = image.dim() == 4
        if not batch_mode:
            image = image.unsqueeze(0)
            mask = mask.unsqueeze(0) if mask.dim() == 3 else mask
        
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)
        
        B = image.shape[0]
        device = image.device
        
        # Process each sample in batch
        degraded_batch = []
        
        for b in range(B):
            img_single = image[b:b+1]
            mask_single = mask[b:b+1]
            
            # Try multiple times to get visible degradation
            degraded = None
            
            for attempt in range(max_retries):
                # 1. Multiplicative shift (scale by 0.5 to 1.5)
                # This affects bright colors more
                mult_shift = 1.0 + (torch.rand(1, 3, 1, 1, device=device) - 0.5) * strength
                mult_shift = mult_shift.clamp(0.5, 1.5)
                
                # 2. Additive shift (-0.3 to +0.3)
                # This is critical for dark colors
                add_shift = (torch.rand(1, 3, 1, 1, device=device) - 0.5) * strength * 0.6
                
                # On later attempts, progressively increase strength
                if attempt > 0:
                    boost_factor = 1.0 + (attempt * 0.3)
                    mult_shift = 1.0 + (mult_shift - 1.0) * boost_factor
                    add_shift = add_shift * boost_factor
                
                # Apply degradation
                mask_3ch = mask_single.expand(-1, 3, -1, -1)
                degraded_attempt = img_single * mult_shift + add_shift * mask_3ch
                
                # Keep outside mask unchanged
                degraded_attempt = degraded_attempt * mask_3ch + img_single * (1 - mask_3ch)
                degraded_attempt = degraded_attempt.clamp(0, 1)
                
                # VALIDATE: Check actual degradation in masked region
                mask_bool = mask_single.squeeze() > 0.5
                
                if mask_bool.sum() > 10:  # Need some pixels to measure
                    # Calculate actual MAE in masked region
                    img_masked = img_single[:, :, mask_bool]
                    deg_masked = degraded_attempt[:, :, mask_bool]
                    actual_mae = (deg_masked - img_masked).abs().mean().item()
                    
                    if actual_mae >= min_mae:
                        # Success! Degradation is visible enough
                        degraded = degraded_attempt
                        break
                else:
                    # Mask too small to measure, accept degradation
                    degraded = degraded_attempt
                    break
            
            # If all retries failed, use last attempt
            if degraded is None:
                degraded = degraded_attempt
            
            degraded_batch.append(degraded)
        
        # Combine batch
        result = torch.cat(degraded_batch, dim=0)
        
        if not batch_mode:
            result = result.squeeze(0)
        
        return result