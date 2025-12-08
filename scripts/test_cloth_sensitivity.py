#!/usr/bin/env python3
"""
Test if model uses cloth reference colors
Tests the production model with different cloth colors
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF

from src.models.fast_unet import MinimalUNet


def load_model(checkpoint_path):
    """Load model from checkpoint"""
    device = torch.device('cpu')
    print(f"💻 Using CPU")
    
    model = MinimalUNet(in_channels=7)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', '?')
        val_loss = checkpoint.get('val_loss', '?')
        print(f"✅ Loaded model from epoch {epoch}, val_loss: {val_loss}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✅ Loaded model weights")
    
    model = model.to(device)
    model.eval()
    return model, device


def load_image(path, size=256):
    """Load image"""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return TF.to_tensor(img)


def create_solid_color_cloth(color_rgb, size=256):
    """Create a solid color cloth reference"""
    img = np.ones((size, size, 3), dtype=np.float32)
    img[:, :, 0] = color_rgb[0] / 255.0
    img[:, :, 1] = color_rgb[1] / 255.0
    img[:, :, 2] = color_rgb[2] / 255.0
    return torch.from_numpy(img).permute(2, 0, 1)


def test_cloth_sensitivity(model, on_model_path, mask_path, device):
    """Test if changing cloth reference changes output"""
    
    print(f"\n📂 Loading test image...")
    
    # Load on-model
    on_model = load_image(on_model_path).to(device)
    
    # Load mask
    mask_pil = Image.open(mask_path)
    if mask_pil.mode == 'P':
        mask = np.array(mask_pil, dtype=np.uint8)
    else:
        mask = np.array(mask_pil.convert('L'), dtype=np.uint8)
    
    mask_pil_resized = Image.fromarray(mask).resize((256, 256), Image.NEAREST)
    mask = np.array(mask_pil_resized, dtype=np.uint8)
    mask = (mask == 5).astype(np.float32)
    mask = torch.from_numpy(mask).unsqueeze(0).to(device)
    
    print(f"   Image: {on_model.shape}")
    print(f"   Mask: {mask.shape}, coverage: {mask.mean()*100:.1f}%")
    
    # Test with different cloth colors
    test_colors = {
        'Red': (255, 0, 0),
        'Green': (0, 255, 0),
        'Blue': (0, 0, 255),
        'Yellow': (255, 255, 0),
        'Cyan': (0, 255, 255),
        'Magenta': (255, 0, 255),
        'White': (255, 255, 255),
        'Orange': (255, 128, 0),
    }
    
    results = {}
    
    print(f"\n🎨 Testing with different cloth colors...")
    
    with torch.no_grad():
        for name, rgb in test_colors.items():
            # Create solid color cloth
            cloth = create_solid_color_cloth(rgb).to(device)
            
            # Apply model
            on_model_batch = on_model.unsqueeze(0)
            cloth_batch = cloth.unsqueeze(0)
            mask_batch = mask.unsqueeze(0)
            
            output = model(on_model_batch, mask_batch, cloth_batch)
            output = output.squeeze(0).cpu()
            
            results[name] = output
            print(f"   ✅ {name}")
    
    # Visualize
    print(f"\n📊 Creating visualization...")
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    def to_img(t):
        return t.permute(1, 2, 0).numpy().clip(0, 1)
    
    # Show input
    axes[0].imshow(to_img(on_model.cpu()))
    axes[0].set_title('Input On-Model\n(Original)', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Show outputs for different cloth colors
    for idx, (name, output) in enumerate(results.items(), start=1):
        axes[idx].imshow(to_img(output))
        
        # Get mean color of output in masked region
        mask_bool = mask.cpu().squeeze() > 0.5
        if mask_bool.any():
            output_masked = output[:, mask_bool]
            mean_color = output_masked.mean(dim=1)
            rgb_str = f"R:{mean_color[0]:.2f} G:{mean_color[1]:.2f} B:{mean_color[2]:.2f}"
        else:
            rgb_str = "No mask"
        
        axes[idx].set_title(f'Cloth: {name}\n{rgb_str}', fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle('PRODUCTION MODEL: Cloth Reference Sensitivity Test\n' +
                 'Outputs should CHANGE with cloth color (color transfer learned)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/cloth_sensitivity_production.png', dpi=150)
    print(f"💾 Saved: outputs/cloth_sensitivity_production.png")
    plt.show()
    
    # Check if outputs are different
    print(f"\n🔍 Analyzing results...")
    outputs_list = list(results.values())
    
    # Compare each output to the first one
    max_diff = 0
    for i in range(1, len(outputs_list)):
        diff = (outputs_list[i] - outputs_list[0]).abs().mean().item()
        max_diff = max(max_diff, diff)
        print(f"   Diff from Red: {list(results.keys())[i]:10s} = {diff:.4f}")
    
    print(f"\n{'='*70}")
    if max_diff < 0.01:
        print("❌ MODEL IGNORES CLOTH REFERENCE")
        print("   All outputs are nearly identical!")
    elif max_diff < 0.05:
        print("⚠️  MODEL PARTIALLY USES CLOTH REFERENCE")
        print("   Some variation, but limited color transfer")
    else:
        print("✅ MODEL USES CLOTH REFERENCE!")
        print("   Outputs change significantly with cloth color!")
        print("   Color transfer learned successfully!")
    print(f"{'='*70}")


if __name__ == '__main__':
    print("="*70)
    print("PRODUCTION MODEL: CLOTH SENSITIVITY TEST")
    print("="*70)
    
    # UPDATE THIS PATH
    checkpoint = 'outputs/cloth_model_lighting_aware/model_best.pth'
    on_model = 'raw_data/test/images/00006_00.jpg'
    mask = 'raw_data/test/image-parse-v3/00006_00.png'
    
    print(f"\n📋 Configuration:")
    print(f"   Checkpoint: {checkpoint}")
    print(f"   Test image: {on_model}")
    print(f"   Mask: {mask}")
    
    print(f"\n🏗️  Loading model...")
    model, device = load_model(checkpoint)
    
    test_cloth_sensitivity(model, on_model, mask, device)
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE!")
    print("="*70)