# -*- coding: utf-8 -*-

#!/usr/bin/env python3

import torch
import torch.nn.functional as F


def simple_structure_loss(output, input_degraded, target, mask):
    """
    Simplified loss - much faster than structure_preserving_loss
    
    The identity skip connection in the model architecture already guarantees
    that outside mask = input, so we don't need complex structure losses.
    
    Args:
        output: [B, 3, H, W] - model output
        input_degraded: [B, 3, H, W] - degraded input (not used, but kept for compatibility)
        target: [B, 3, H, W] - ground truth
        mask: [B, 1, H, W] - garment mask
    
    Returns:
        loss: scalar tensor
        components: dict with individual loss components
    """
    mask_3ch = mask.expand(-1, 3, -1, -1)
    
    # 1. Global reconstruction loss
    recon_loss = F.mse_loss(output, target)
    
    # 2. Inside mask: should match target strongly
    inside_loss = F.mse_loss(
        output * mask_3ch,
        target * mask_3ch
    )
    
    # That's it! Simple and fast.
    # Identity skip guarantees outside mask matches input.
    
    total_loss = recon_loss + 2.0 * inside_loss
    
    return total_loss, {
        'recon': recon_loss.item(),
        'inside': inside_loss.item(),
    }