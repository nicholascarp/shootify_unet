#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:26:32 2025

@author: nicholascarp
"""

"""
Loss functions for color correction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ColorCorrectionLoss(nn.Module):
    """
    Combined loss for color correction:
    1. Global MSE: Overall color accuracy
    2. Masked MSE: Focused on garment region (weighted)
    """
    
    def __init__(self, mask_weight=2.0):
        """
        Args:
            mask_weight: Weight multiplier for masked region loss (default 2.0)
        """
        super().__init__()
        self.mask_weight = mask_weight
    
    def forward(self, pred, target, mask):
        """
        Compute combined loss
        
        Args:
            pred: [B, 3, H, W] - Predicted corrected image
            target: [B, 3, H, W] - Ground truth image
            mask: [B, 1, H, W] - Binary mask for garment region
        
        Returns:
            loss: Scalar tensor with combined loss
            loss_global: Global MSE loss value (for logging)
            loss_masked: Masked MSE loss value (for logging)
        """
        # Global loss (entire image)
        loss_global = F.mse_loss(pred, target)
        
        # Masked loss (garment region only)
        mask_3ch = mask.expand(-1, 3, -1, -1)
        pred_masked = pred * mask_3ch
        target_masked = target * mask_3ch
        loss_masked = F.mse_loss(pred_masked, target_masked)
        
        # Combined
        total_loss = loss_global + self.mask_weight * loss_masked
        
        return total_loss, loss_global.item(), loss_masked.item()