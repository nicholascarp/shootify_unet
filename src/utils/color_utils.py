"""
Color extraction and processing utilities
"""

import torch


def extract_color_conditioning(image, mask):
    """
    Extract average color from masked region for conditioning
    
    Args:
        image: [B, 3, H, W] - Image tensor
        mask: [B, 1, H, W] - Binary mask
    
    Returns:
        color_cond: [B, 3, H, W] - Spatially expanded color conditioning
    """
    # Expand mask to 3 channels
    mask_3ch = mask.expand(-1, 3, -1, -1)
    
    # Compute average color in masked region
    color = (image * mask_3ch).sum(dim=(2, 3)) / (mask_3ch.sum(dim=(2, 3)) + 1e-6)
    
    # Expand to spatial dimensions
    color_cond = color[:, :, None, None].expand_as(image)
    
    return color_cond
