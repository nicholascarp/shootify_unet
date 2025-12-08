"""
Evaluation metrics for color correction
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_color_accuracy(pred, target, mask):
    """
    Compute color accuracy in masked region (normalized by mask area)
    
    Args:
        pred: [B, 3, H, W] - Predicted image
        target: [B, 3, H, W] - Target image
        mask: [B, 1, H, W] - Binary mask
    
    Returns:
        Mean absolute color difference per pixel in masked region
    """
    mask_3ch = mask.expand(-1, 3, -1, -1)
    
    # NEW: Compute error without averaging
    color_diff = (pred * mask_3ch - target * mask_3ch).abs()
    
    # NEW: Normalize by mask area only
    mask_area = mask_3ch.sum() + 1e-8  # Avoid division by zero
    color_mae = color_diff.sum() / mask_area  # ✅ Divides by mask pixels only
    
    return color_mae.item()

def compute_mse(pred, target, mask=None):
    """
    Compute MSE (optionally masked and normalized by mask area)
    
    Args:
        pred: [B, 3, H, W] - Predicted image
        target: [B, 3, H, W] - Target image
        mask: [B, 1, H, W] - Optional mask
    
    Returns:
        MSE value (normalized by mask area if mask provided)
    """
    if mask is not None:
        mask_3ch = mask.expand(-1, 3, -1, -1)
        
        # NEW: Compute squared error (don't zero out inputs)
        squared_error = ((pred - target) * mask_3ch).pow(2)
        
        # NEW: Normalize by mask area only
        mask_area = mask_3ch.sum() + 1e-8
        mse = squared_error.sum() / mask_area  # ✅ Divides by mask pixels only
        
        return mse.item()
    else:
        # Global MSE (no normalization needed)
        return F.mse_loss(pred, target).item()

def compute_psnr(pred, target, mask=None):
    """
    Compute PSNR (Peak Signal-to-Noise Ratio)
    
    Args:
        pred: [B, 3, H, W] - Predicted image in [0, 1]
        target: [B, 3, H, W] - Target image in [0, 1]
        mask: [B, 1, H, W] - Optional mask
    
    Returns:
        PSNR value in dB
    """
    mse = compute_mse(pred, target, mask)
    if mse < 1e-10:
        return 100.0
    return 20 * np.log10(1.0 / np.sqrt(mse))


def compute_metrics(pred, target, mask):
    """
    Compute all metrics
    
    Args:
        pred: [B, 3, H, W] - Predicted image
        target: [B, 3, H, W] - Target image
        mask: [B, 1, H, W] - Binary mask
    
    Returns:
        Dictionary with all metrics
    """
    return {
        'color_accuracy': compute_color_accuracy(pred, target, mask),
        'mse_global': compute_mse(pred, target, mask=None),
        'mse_masked': compute_mse(pred, target, mask=mask),
        'psnr_global': compute_psnr(pred, target, mask=None),
        'psnr_masked': compute_psnr(pred, target, mask=mask)
    }
