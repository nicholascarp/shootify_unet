#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:28:35 2025

@author: nicholascarp
"""

#!/usr/bin/env python3
#!/usr/bin/env python3
"""
Inference script for color correction model
"""

import os
import sys
from pathlib import Path

# ADD SRC TO PATH FIRST - BEFORE ANY SRC IMPORTS
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch import from_numpy
from PIL import Image

# NOW import from src (after path is set)
from src.models import FastColorCorrectionUNet
from src.data import ColorDegradation
from src.utils import extract_color_conditioning, visualize_results


def load_model(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    model = FastColorCorrectionUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config


def load_image(image_path, img_size):
    """Load and preprocess image"""
    from pathlib import Path
    
    image_path = Path(image_path)  # ← ADD THIS
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to tensor [3, H, W] in range [0, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)  # HWC -> CHW
    
    return img_tensor.unsqueeze(0)  # Add batch dimension


def load_mask(mask_path, img_size):
    """Load and preprocess mask"""
    from pathlib import Path
    
    mask_path = Path(mask_path)
    
    # Load mask
    if mask_path.suffix == '.npy':
        mask = np.load(mask_path)
    elif mask_path.suffix in ['.png', '.jpg', '.jpeg']:
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask) > 128
    else:
        raise ValueError(f"Unsupported mask format: {mask_path.suffix}")
    
    # Convert to numpy and ensure 2D
    mask = np.array(mask, dtype=np.float32)
    while mask.ndim > 2:
        mask = np.squeeze(mask, axis=0)
    
    # To tensor [H, W] -> [1, 1, H, W]
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    
    # Resize
    if mask_tensor.shape[-2:] != (img_size, img_size):
        mask_tensor = F.interpolate(mask_tensor, size=(img_size, img_size), mode='nearest')
    
    return mask_tensor


def correct_colors(model, degraded, mask, reference, device):
    """Run color correction inference"""
    # Move to device
    degraded = degraded.to(device)
    mask = mask.to(device)
    reference = reference.to(device)
    
    # Ensure mask is exactly 4D: [B, C, H, W]
    while mask.ndim > 4:
        mask = mask.squeeze(0)
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(0)
    
    # Extract color conditioning
    color_cond = extract_color_conditioning(reference, mask)
    
    # Forward pass
    with torch.no_grad():
        output = model(degraded, mask, color_cond)
    
    return output


def main(args):
    print("="*70)
    print("SHOOTIFY COLOR CORRECTION - INFERENCE")
    print("="*70)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Load model
    print(f"\n🔧 Loading model...")
    model, config = load_model(args.checkpoint, device)
    img_size = config['training']['img_size']
    print(f"✅ Model loaded from: {args.checkpoint}")
    print(f"   Image size: {img_size}x{img_size}")
    
    # Load inputs
    print(f"✅ Image: {args.degraded}")
    print(f"✅ Mask: {args.mask}")
    print(f"✅ Reference: {args.reference}")

    # Load original image and mask
    original = load_image(args.degraded, img_size)
    mask = load_mask(args.mask, img_size)
    reference = load_image(args.reference, img_size)

    # Apply degradation (like during training)
    print(f"\n🎨 Applying artificial degradation...")

    # Remove batch dimension for degradation [1,3,H,W] -> [3,H,W]
    original_no_batch = original.squeeze(0)
    mask_no_batch = mask.squeeze(0)

    # Apply degradation (static method - no instance needed)
    degraded_no_batch = ColorDegradation.apply_random_degradation(
        original_no_batch, 
        mask_no_batch, 
        strength=0.3
    )

    # Add batch dimension back [3,H,W] -> [1,3,H,W]
    degraded = degraded_no_batch.unsqueeze(0)

    print(f"✅ Degradation applied")
    
    # Run inference
    print(f"\n🎨 Correcting colors...")
    corrected = correct_colors(model, degraded, mask, reference, device)
    print(f"✅ Color correction complete")
    
    # Convert to numpy for visualization first (keep in [0,1] range as float)
    corrected_np_float = corrected.squeeze(0).cpu().permute(1, 2, 0).numpy()
    
    # Convert to uint8 ONLY for saving the output file
    corrected_np_uint8 = (corrected_np_float * 255).clip(0, 255).astype(np.uint8)
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(corrected_np_uint8).save(output_path)
    print(f"\n💾 Output saved to: {output_path}")
    
    # Create visualization if requested
    if args.visualize:
        viz_path = output_path.parent / f"{output_path.stem}_visualization.png"
        
        # Prepare images for visualization (remove batch dims)
        degraded_np = degraded.squeeze(0).cpu().permute(1, 2, 0).numpy()
        reference_np = reference.squeeze(0).cpu().permute(1, 2, 0).numpy()
        mask_np = mask.cpu().numpy().squeeze()  # Squeeze all dimensions

        # Ensure mask is 2D
        while mask_np.ndim > 2:
            mask_np = mask_np[0]

        # Prepare original image too
        original_np = original.squeeze(0).cpu().permute(1, 2, 0).numpy()

        visualize_results(
            from_numpy(original_np).permute(2, 0, 1),          # Original (clean)
            from_numpy(degraded_np).permute(2, 0, 1),          # Degraded (with color shift)
            from_numpy(corrected_np_float).permute(2, 0, 1),   # Corrected - FIXED: Use float version!
            from_numpy(mask_np),
            save_path=viz_path
        )
    
    print(f"\n{'='*70}")
    print("✅ INFERENCE COMPLETE")
    print(f"{'='*70}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on new images')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--degraded', type=str, required=True, help='Path to degraded input image')
    parser.add_argument('--mask', type=str, required=True, help='Path to garment mask (.npy or image)')
    parser.add_argument('--reference', type=str, required=True, help='Path to reference image (for color)')
    parser.add_argument('--output', type=str, required=True, help='Path to save corrected output')
    parser.add_argument('--visualize', action='store_true', help='Create visualization of results')
    
    args = parser.parse_args()
    main(args)
