#!/usr/bin/env python3
"""
FIXED: Batch inference with proper lighting-aware cloth preprocessing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.models.fast_unet import MinimalUNet
from src.data.degradation import ColorDegradation


def load_model(checkpoint_path):
    """Load model"""
    device = torch.device('cpu')
    model = MinimalUNet(in_channels=7)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, device


def load_image(path, size=256):
    """Load and resize image"""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return TF.to_tensor(img)


def load_mask(path, size=256):
    """Load mask"""
    mask_pil = Image.open(path)
    
    if mask_pil.mode == 'P':
        mask = np.array(mask_pil, dtype=np.uint8)
    else:
        mask = np.array(mask_pil.convert('L'), dtype=np.uint8)
    
    if mask.shape[0] != size or mask.shape[1] != size:
        mask_pil_resized = Image.fromarray(mask)
        mask_pil_resized = mask_pil_resized.resize((size, size), Image.NEAREST)
        mask = np.array(mask_pil_resized, dtype=np.uint8)
    
    mask = (mask == 5).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0)
    
    return mask


def create_lighting_aware_cloth(cloth: torch.Tensor, on_model: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    CRITICAL: Apply the same lighting-aware preprocessing as training!
    """
    # 1. Extract mean color from cloth
    cloth_mean_color = cloth.mean(dim=[1, 2])  # [3]
    
    # 2. Extract luminance from on-model garment
    on_model_luminance = (
        0.299 * on_model[0] + 
        0.587 * on_model[1] + 
        0.114 * on_model[2]
    )
    
    # 3. Normalize luminance in masked region
    mask_bool = mask.squeeze(0) > 0.5
    
    if mask_bool.sum() > 0:
        lum_values = on_model_luminance[mask_bool]
        lum_min = lum_values.min()
        lum_max = lum_values.max()
        
        if lum_max > lum_min:
            normalized_lum = (on_model_luminance - lum_min) / (lum_max - lum_min + 1e-6)
            normalized_lum = normalized_lum * 1.0 + 0.3
        else:
            normalized_lum = torch.ones_like(on_model_luminance)
    else:
        normalized_lum = torch.ones_like(on_model_luminance)
    
    # 4. Apply cloth color with lighting pattern
    cloth_with_lighting = cloth_mean_color.view(3, 1, 1) * normalized_lum.unsqueeze(0)
    cloth_with_lighting = cloth_with_lighting.clamp(0, 1)
    
    return cloth_with_lighting


def test_batch_inference(model, device, test_dir, num_samples=10):
    """Test model on multiple samples with CORRECT preprocessing"""
    
    print("="*70)
    print("FIXED BATCH INFERENCE TEST - With Lighting-Aware Cloth")
    print("="*70)
    
    image_dir = os.path.join(test_dir, 'images')
    cloth_dir = os.path.join(test_dir, 'cloth')
    mask_dir = os.path.join(test_dir, 'image-parse-v3')
    
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])[:num_samples]
    
    print(f"\n Testing {len(image_files)} samples...")
    
    fig, axes = plt.subplots(len(image_files), 7, figsize=(28, 4*len(image_files)))
    
    def to_img(t):
        return t.permute(1, 2, 0).numpy().clip(0, 1)
    
    for idx, img_file in enumerate(image_files):
        base_name = img_file.replace('.jpg', '')
        
        on_model_path = os.path.join(image_dir, img_file)
        cloth_path = os.path.join(cloth_dir, img_file)
        mask_path = os.path.join(mask_dir, base_name + '.png')
        
        on_model = load_image(on_model_path)
        cloth_raw = load_image(cloth_path)
        mask = load_mask(mask_path)
        
        # CRITICAL: Create lighting-aware cloth (same as training!)
        cloth_lighting_aware = create_lighting_aware_cloth(cloth_raw, on_model, mask)
        
        # Apply degradation
        degraded = ColorDegradation.apply_realistic_degradation(
            on_model, mask, strength=0.5
        )
        
        mask_bool = mask.squeeze() > 0.5
        if mask_bool.sum() > 0:
            deg_amount = (degraded[:, mask_bool] - on_model[:, mask_bool]).abs().mean().item()
        else:
            deg_amount = 0
        
        # Apply correction with lighting-aware cloth
        with torch.no_grad():
            degraded_batch = degraded.unsqueeze(0).to(device)
            cloth_batch = cloth_lighting_aware.unsqueeze(0).to(device)
            mask_batch = mask.unsqueeze(0).to(device)
            
            corrected = model(degraded_batch, mask_batch, cloth_batch)
            corrected = corrected.squeeze(0).cpu()
        
        if mask_bool.sum() > 0:
            final_error = (corrected[:, mask_bool] - on_model[:, mask_bool]).abs().mean().item()
            correction_improvement = ((deg_amount - final_error) / deg_amount * 100) if deg_amount > 0 else 0
        else:
            final_error = 0
            correction_improvement = 0
        
        # Plot
        axes[idx, 0].imshow(to_img(on_model))
        axes[idx, 0].set_title(f'Sample {idx+1}\nOriginal', fontsize=9)
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(to_img(cloth_raw))
        axes[idx, 1].set_title('Raw Cloth\n(Not used)', fontsize=9, color='red')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(to_img(cloth_lighting_aware))
        axes[idx, 2].set_title('Lighting-Aware\n(Actually used)', fontsize=9, color='green')
        axes[idx, 2].axis('off')
        
        axes[idx, 3].imshow(to_img(degraded))
        axes[idx, 3].set_title(f'Degraded\n(MAE: {deg_amount:.3f})', fontsize=9)
        axes[idx, 3].axis('off')
        
        axes[idx, 4].imshow(mask.squeeze().numpy(), cmap='gray')
        axes[idx, 4].set_title(f'Mask\n({mask.mean()*100:.1f}%)', fontsize=9)
        axes[idx, 4].axis('off')
        
        axes[idx, 5].imshow(to_img(corrected))
        axes[idx, 5].set_title(f'Corrected\n(MAE: {final_error:.3f})', fontsize=9)
        axes[idx, 5].axis('off')
        
        error_vis = (corrected - on_model).abs() * 5
        
        if correction_improvement > 50:
            color = 'green'
            status = 'Good'
        elif correction_improvement > 0:
            color = 'orange'
            status = 'Partial'
        else:
            color = 'red'
            status = 'Failed'
        
        axes[idx, 6].imshow(to_img(error_vis))
        axes[idx, 6].set_title(f'Error (5x)\n{status}: {correction_improvement:.0f}%', 
                               fontsize=9, color=color, fontweight='bold')
        axes[idx, 6].axis('off')
        
        print(f"   Sample {idx+1}: Degradation={deg_amount:.3f}, "
              f"Final error={final_error:.3f}, "
              f"Improvement={correction_improvement:.0f}% - {status}")
    
    plt.suptitle('FIXED Batch Inference - Now Using Lighting-Aware Cloth!\n' + 
                 'Shows: Original → Raw Cloth → Lighting-Aware Cloth → Degrade → Correct',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = 'outputs/batch_inference_FIXED.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n Saved: {output_path}")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/cloth_model_lighting_aware/model_best.pth')
    parser.add_argument('--test-dir', type=str, default='raw_data/test')
    parser.add_argument('--num-samples', type=int, default=15)
    
    args = parser.parse_args()
    
    model, device = load_model(args.checkpoint)
    test_batch_inference(model, device, args.test_dir, args.num_samples)