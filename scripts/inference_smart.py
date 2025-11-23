#!/usr/bin/env python3
"""
Smart inference script with flexible modes:
- Mode 1: Apply degradation then correct (for testing with clean images)
- Mode 2: Direct correction (for already degraded images with reference)
"""

import os
import sys
from pathlib import Path

# ADD SRC TO PATH FIRST
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

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
    image_path = Path(image_path)
    
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    
    # Convert to tensor [3, H, W] in range [0, 1]
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1)
    
    return img_tensor.unsqueeze(0)  # Add batch dimension


def load_mask(mask_path, img_size):
    """Load and preprocess mask"""
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


def apply_degradation(image, mask, strength=0.5):
    """
    Apply artificial color degradation to image
    
    Args:
        image: [1, 3, H, W] tensor
        mask: [1, 1, H, W] tensor
        strength: Degradation strength
    
    Returns:
        Degraded image [1, 3, H, W]
    """
    # Remove batch dimension for degradation [1,3,H,W] -> [3,H,W]
    image_no_batch = image.squeeze(0)
    mask_no_batch = mask.squeeze(0)
    
    # Apply degradation
    degraded_no_batch = ColorDegradation.apply_random_degradation(
        image_no_batch, 
        mask_no_batch, 
        strength=strength
    )
    
    # Add batch dimension back [3,H,W] -> [1,3,H,W]
    return degraded_no_batch.unsqueeze(0)


def correct_colors(model, degraded, mask, reference, device):
    """
    Run color correction inference
    
    Args:
        model: Trained model
        degraded: [1, 3, H, W] tensor - Image to correct
        mask: [1, 1, H, W] tensor - Garment mask
        reference: [1, 3, H, W] tensor - Reference for color
        device: torch.device
    
    Returns:
        Corrected image [1, 3, H, W]
    """
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
    
    # Extract color conditioning from reference
    color_cond = extract_color_conditioning(reference, mask)
    
    # Forward pass
    with torch.no_grad():
        corrected = model(degraded, mask, color_cond)
    
    return corrected


def save_image(tensor, path):
    """
    Save tensor as image
    
    Args:
        tensor: [1, 3, H, W] or [3, H, W] tensor
        path: Output path
    """
    # Remove batch dimension if present
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    
    # Convert to numpy HWC
    img_np = tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    
    # Save
    Image.fromarray(img_np).save(path)


def mode_1_degrade_and_correct(model, clean_image, mask, reference, device, degradation_strength):
    """
    Mode 1: Apply degradation first, then correct
    Use this for testing with clean images
    
    Returns:
        (degraded, corrected) tensors
    """
    print("📋 Mode 1: Apply degradation → Correct colors")
    print("   (Testing mode: simulates color shift then corrects it)")
    
    # Apply degradation
    print(f"   🎨 Applying artificial degradation (strength={degradation_strength})...")
    degraded = apply_degradation(clean_image, mask, strength=degradation_strength)
    
    # Correct colors
    print(f"   🔧 Correcting colors...")
    corrected = correct_colors(model, degraded, mask, reference, device)
    
    return degraded, corrected


def mode_2_direct_correction(model, degraded_image, mask, reference, device):
    """
    Mode 2: Direct correction without degradation
    Use this for already degraded images (e.g., from generative AI)
    
    Returns:
        corrected tensor
    """
    print("📋 Mode 2: Direct color correction")
    print("   (Production mode: corrects existing color issues)")
    
    # Direct correction
    print(f"   🔧 Correcting colors...")
    corrected = correct_colors(model, degraded_image, mask, reference, device)
    
    return corrected


def main(args):
    print("="*70)
    print("SHOOTIFY COLOR CORRECTION - SMART INFERENCE")
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
    print(f"\n📂 Loading inputs...")
    print(f"   Image: {args.image}")
    print(f"   Mask: {args.mask}")
    print(f"   Reference: {args.reference}")
    
    image = load_image(args.image, img_size)
    mask = load_mask(args.mask, img_size)
    reference = load_image(args.reference, img_size)
    
    print(f"✅ Inputs loaded")
    
    # Determine mode
    if args.mode == 'auto':
        # Auto-detect: if image and reference are the same file, use mode 1
        if Path(args.image).resolve() == Path(args.reference).resolve():
            mode = 'degrade-and-correct'
            print(f"\n🤖 Auto-detected mode: degrade-and-correct")
            print(f"   (Image and reference are the same)")
        else:
            mode = 'direct'
            print(f"\n🤖 Auto-detected mode: direct")
            print(f"   (Image and reference are different)")
    else:
        mode = args.mode
        print(f"\n📋 Using specified mode: {mode}")
    
    # Run inference based on mode
    print(f"\n🚀 Running inference...\n")
    
    if mode == 'degrade-and-correct':
        # Mode 1: Testing mode
        degraded, corrected = mode_1_degrade_and_correct(
            model=model,
            clean_image=image,
            mask=mask,
            reference=reference,
            device=device,
            degradation_strength=args.degradation_strength
        )
        
        # Save degraded intermediate
        if args.save_degraded:
            degraded_path = Path(args.output).parent / f"{Path(args.output).stem}_degraded.jpg"
            save_image(degraded, degraded_path)
            print(f"\n💾 Degraded image saved to: {degraded_path}")
        
    else:
        # Mode 2: Production mode
        corrected = mode_2_direct_correction(
            model=model,
            degraded_image=image,
            mask=mask,
            reference=reference,
            device=device
        )
        degraded = image  # For visualization
    
    print(f"✅ Inference complete")
    
    # Save corrected output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(corrected, output_path)
    print(f"\n💾 Corrected image saved to: {output_path}")
    
    # Create visualization if requested
    if args.visualize:
        viz_path = output_path.parent / f"{output_path.stem}_visualization.png"
        
        print(f"\n🎨 Creating visualization...")
        
        # Prepare images for visualization
        original_np = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
        degraded_np = degraded.squeeze(0).cpu().permute(1, 2, 0).numpy()
        corrected_np = corrected.squeeze(0).cpu().permute(1, 2, 0).numpy()
        mask_np = mask.cpu().numpy().squeeze()
        
        # Ensure mask is 2D
        while mask_np.ndim > 2:
            mask_np = mask_np[0]
        
        visualize_results(
            torch.from_numpy(original_np).permute(2, 0, 1),
            torch.from_numpy(degraded_np).permute(2, 0, 1),
            torch.from_numpy(corrected_np).permute(2, 0, 1),
            torch.from_numpy(mask_np),
            save_path=viz_path
        )
        
        print(f"✅ Visualization saved to: {viz_path}")
    
    print(f"\n{'='*70}")
    print("✅ INFERENCE COMPLETE")
    print(f"{'='*70}")
    
    # Print mode summary
    print(f"\n📊 Summary:")
    print(f"   Mode: {mode}")
    if mode == 'degrade-and-correct':
        print(f"   Applied degradation: Yes (strength={args.degradation_strength})")
    else:
        print(f"   Applied degradation: No (direct correction)")
    print(f"   Output: {output_path}")
    if args.visualize:
        print(f"   Visualization: {viz_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smart inference with flexible modes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  degrade-and-correct : Apply artificial degradation, then correct (testing)
  direct             : Direct correction without degradation (production)
  auto               : Automatically detect based on inputs (default)

Examples:
  # Testing mode: clean image → degrade → correct
  python scripts/inference_smart.py \\
      --checkpoint model.pth \\
      --image clean.jpg \\
      --mask mask.npy \\
      --reference clean.jpg \\
      --output corrected.jpg \\
      --mode degrade-and-correct

  # Production mode: already degraded image → correct
  python scripts/inference_smart.py \\
      --checkpoint model.pth \\
      --image degraded.jpg \\
      --mask mask.npy \\
      --reference reference.jpg \\
      --output corrected.jpg \\
      --mode direct

  # Auto mode: detects mode based on whether image and reference are same
  python scripts/inference_smart.py \\
      --checkpoint model.pth \\
      --image image.jpg \\
      --mask mask.npy \\
      --reference reference.jpg \\
      --output corrected.jpg
        """
    )
    
    # Required arguments
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True, 
                       help='Input image (clean or degraded)')
    parser.add_argument('--mask', type=str, required=True, 
                       help='Garment mask (.npy or image)')
    parser.add_argument('--reference', type=str, required=True, 
                       help='Reference image for color (still-life)')
    parser.add_argument('--output', type=str, required=True, 
                       help='Output path for corrected image')
    
    # Mode selection
    parser.add_argument('--mode', type=str, 
                       choices=['degrade-and-correct', 'direct', 'auto'],
                       default='auto',
                       help='Inference mode (default: auto)')
    
    # Optional arguments
    parser.add_argument('--degradation-strength', type=float, default=0.5,
                       help='Degradation strength for mode 1 (default: 0.5)')
    parser.add_argument('--save-degraded', action='store_true',
                       help='Save intermediate degraded image (mode 1 only)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization of results')
    
    args = parser.parse_args()
    main(args)
