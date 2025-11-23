"""
Evaluation metrics for color correction
"""

import torch
import torch.nn.functional as F
import numpy as np


def compute_color_accuracy(pred, target, mask):
    """
    Compute color accuracy in masked region
    
    Args:
        pred: [B, 3, H, W] - Predicted image
        target: [B, 3, H, W] - Target image
        mask: [B, 1, H, W] - Binary mask
    
    Returns:
        Mean absolute color difference in masked region
    """
    mask_3ch = mask.expand(-1, 3, -1, -1)
    color_diff = (pred * mask_3ch - target * mask_3ch).abs().mean()
    return color_diff.item()


def compute_mse(pred, target, mask=None):
    """
    Compute MSE (optionally masked)
    
    Args:
        pred: [B, 3, H, W] - Predicted image
        target: [B, 3, H, W] - Target image
        mask: [B, 1, H, W] - Optional mask
    
    Returns:
        MSE value
    """
    if mask is not None:
        mask_3ch = mask.expand(-1, 3, -1, -1)
        pred = pred * mask_3ch
        target = target * mask_3ch
    
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
