#!/usr/bin/env python3
"""
Test model on external ATR dataset from HuggingFace
DIFFERENT source from VITON-HD + PROPER segmentation masks!
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm

from src.models.fast_unet import MinimalUNet
from src.data.degradation import ColorDegradation


def create_lighting_aware_cloth(cloth, on_model, mask):
    """Apply lighting-aware preprocessing"""
    cloth_mean_color = cloth.mean(dim=[1, 2])
    on_model_luminance = 0.299 * on_model[0] + 0.587 * on_model[1] + 0.114 * on_model[2]
    
    mask_bool = mask.squeeze(0) > 0.5
    if mask_bool.sum() > 0:
        lum_values = on_model_luminance[mask_bool]
        lum_min, lum_max = lum_values.min(), lum_values.max()
        
        if lum_max > lum_min:
            normalized_lum = (on_model_luminance - lum_min) / (lum_max - lum_min + 1e-6)
            normalized_lum = normalized_lum * 1.0 + 0.3
        else:
            normalized_lum = torch.ones_like(on_model_luminance)
    else:
        normalized_lum = torch.ones_like(on_model_luminance)
    
    cloth_with_lighting = cloth_mean_color.view(3, 1, 1) * normalized_lum.unsqueeze(0)
    return cloth_with_lighting.clamp(0, 1)


def pil_to_tensor(img, size=256):
    """Convert PIL image to tensor"""
    img = img.convert('RGB').resize((size, size), Image.BILINEAR)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def extract_upper_mask_from_segmentation(seg_img, size=256):
    """
    Extract upper clothes mask from ATR segmentation
    Label 4 = Upper-clothes (SAME as VITON-HD!)
    """
    seg_img = seg_img.resize((size, size), Image.NEAREST)
    seg_array = np.array(seg_img, dtype=np.uint8)
    mask = (seg_array == 4).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


def main():
    print("="*70)
    print("EXTERNAL DATASET TEST - ATR Dataset (HuggingFace)")
    print("="*70)
    print("✅ DIFFERENT source: ATR (fashion dataset) vs VITON-HD (Zalando)")
    print("✅ PROPER masks: Label 4 = Upper-clothes")
    
    # Download dataset
    print("\n📥 Downloading ATR dataset from HuggingFace...")
    print("   Dataset: mattmdjaga/human_parsing_dataset")
    dataset = load_dataset("mattmdjaga/human_parsing_dataset", split="train")
    print(f"✅ Loaded {len(dataset)} samples with segmentation masks")
    
    # Inspect dataset structure
    print("\n🔍 Inspecting dataset structure...")
    sample = dataset[0]
    print(f"   Available keys: {list(sample.keys())}")
    for key in sample.keys():
        value = sample[key]
        if isinstance(value, Image.Image):
            print(f"   - {key}: PIL Image {value.size}")
        else:
            print(f"   - {key}: {type(value)}")
    
    # Determine correct key for segmentation mask
    # Common possibilities: 'label', 'mask', 'segmentation', 'annotation', 'labels'
    mask_key = None
    for possible_key in ['label', 'mask', 'segmentation', 'annotation', 'labels', 'parse']:
        if possible_key in sample.keys():
            mask_key = possible_key
            print(f"   ✅ Found mask key: '{mask_key}'")
            break
    
    if mask_key is None:
        print("\n❌ ERROR: Could not find segmentation mask key!")
        print("   Available keys:", list(sample.keys()))
        print("\n   Please check one of the images to identify the mask:")
        for key in sample.keys():
            if isinstance(sample[key], Image.Image):
                print(f"   - Showing {key}...")
                # sample[key].show()  # Uncomment if you want to see images
        return
    
    # Load model
    print("\n🏗️  Loading model...")
    device = torch.device('cpu')
    model = MinimalUNet(in_channels=7)
    checkpoint = torch.load('outputs/cloth_model_lighting_aware/model_best.pth', 
                           map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ Model loaded (trained on VITON-HD/Zalando)")
    
    # Filter samples with upper-clothes
    print(f"\n🔍 Filtering samples with upper-clothes (label 4 in '{mask_key}')...")
    valid_samples = []
    for idx in tqdm(range(len(dataset)), desc="Scanning"):
        seg = np.array(dataset[idx][mask_key])
        if (seg == 4).sum() > 100:  # At least 100 pixels of upper-clothes
            valid_samples.append(idx)
        if len(valid_samples) >= 500:  # Get enough samples
            break
    
    print(f"✅ Found {len(valid_samples)} samples with upper-clothes")
    
    if len(valid_samples) == 0:
        print("\n❌ ERROR: No samples found with upper-clothes (label 4)!")
        print("   Checking what labels exist in first 10 samples...")
        for i in range(min(10, len(dataset))):
            seg = np.array(dataset[i][mask_key])
            unique_labels = np.unique(seg)
            print(f"   Sample {i}: labels = {unique_labels}")
        return
    
    # Test on random samples
    num_samples = min(15, len(valid_samples))
    test_indices = np.random.choice(valid_samples, num_samples, replace=False)
    
    print(f"\n🔍 Testing on {num_samples} random ATR samples...")
    print("   Note: Using same image as cloth reference (self-correction test)")
    
    results = []
    
    # Determine image key
    image_key = 'image' if 'image' in dataset[0].keys() else list(dataset[0].keys())[0]
    print(f"   Using image key: '{image_key}'")
    
    for idx in tqdm(test_indices):
        sample = dataset[int(idx)]
        
        # Get image and segmentation
        on_model = pil_to_tensor(sample[image_key])
        cloth_raw = on_model.clone()
        
        # Extract PROPER upper-clothes mask (label 4)
        mask = extract_upper_mask_from_segmentation(sample[mask_key])
        
        # Skip if mask is too small
        if mask.mean() < 0.05:
            continue
        
        # Create lighting-aware cloth
        cloth = create_lighting_aware_cloth(cloth_raw, on_model, mask)
        
        # Apply degradation
        degraded = ColorDegradation.apply_realistic_degradation(
            on_model, mask, strength=0.5
        )
        
        # Inference
        with torch.no_grad():
            output = model(
                degraded.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
                cloth.unsqueeze(0).to(device)
            ).squeeze(0).cpu()
        
        # Calculate metrics
        mask_bool = mask.squeeze() > 0.5
        if mask_bool.sum() > 0:
            deg_mae = (degraded[:, mask_bool] - on_model[:, mask_bool]).abs().mean().item()
            final_mae = (output[:, mask_bool] - on_model[:, mask_bool]).abs().mean().item()
            improvement = ((deg_mae - final_mae) / deg_mae * 100) if deg_mae > 0 else 0
        else:
            deg_mae, final_mae, improvement = 0, 0, 0
        
        results.append({
            'original': on_model,
            'cloth': cloth,
            'degraded': degraded,
            'output': output,
            'mask': mask,
            'deg_mae': deg_mae,
            'final_mae': final_mae,
            'improvement': improvement
        })
    
    if len(results) == 0:
        print("\n❌ ERROR: No valid results generated!")
        return
    
    # Visualize
    print("\n📊 Creating visualization...")
    fig, axes = plt.subplots(len(results), 6, figsize=(24, 4*len(results)))
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    def to_img(t):
        return t.permute(1, 2, 0).numpy().clip(0, 1)
    
    for i, res in enumerate(results):
        axes[i, 0].imshow(to_img(res['original']))
        axes[i, 0].set_title(f'Sample {i+1}\nOriginal (ATR)', fontsize=9)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(to_img(res['cloth']))
        axes[i, 1].set_title('Lighting-Aware\nCloth', fontsize=9)
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(to_img(res['degraded']))
        axes[i, 2].set_title(f'Degraded\n(MAE: {res["deg_mae"]:.3f})', fontsize=9)
        axes[i, 2].axis('off')
        
        axes[i, 3].imshow(res['mask'].squeeze().numpy(), cmap='gray')
        axes[i, 3].set_title(f'Real Mask\n(label=4, {res["mask"].mean()*100:.1f}%)', fontsize=9)
        axes[i, 3].axis('off')
        
        axes[i, 4].imshow(to_img(res['output']))
        axes[i, 4].set_title(f'Corrected\n(MAE: {res["final_mae"]:.3f})', fontsize=9)
        axes[i, 4].axis('off')
        
        error = (res['output'] - res['original']).abs() * 5
        status = 'Good' if res['improvement'] > 50 else 'Partial' if res['improvement'] > 0 else 'Failed'
        color = 'green' if res['improvement'] > 50 else 'orange' if res['improvement'] > 0 else 'red'
        
        axes[i, 5].imshow(to_img(error))
        axes[i, 5].set_title(f'Error (5x)\n{status}: {res["improvement"]:.0f}%', 
                            fontsize=9, color=color, fontweight='bold')
        axes[i, 5].axis('off')
    
    plt.suptitle('✅ VALID External Test - ATR Dataset (HuggingFace)\n' +
                 'Trained: VITON-HD (Zalando) | Tested: ATR (Fashion Dataset)\n' +
                 'PROPER segmentation masks (label 4 = upper-clothes)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/external_test_atr.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: outputs/external_test_atr.png")
    
    # Summary
    print("\n" + "="*70)
    print("📊 EXTERNAL DATASET TEST SUMMARY")
    print("="*70)
    avg_improvement = np.mean([r['improvement'] for r in results])
    success_rate = sum(1 for r in results if r['improvement'] > 50) / len(results) * 100
    
    print(f"\n✅ Dataset: ATR (DIFFERENT from VITON-HD training data)")
    print(f"✅ Masks: PROPER segmentation (label 4 = upper-clothes)")
    print(f"\nAverage improvement: {avg_improvement:.1f}%")
    print(f"Success rate (>50%): {success_rate:.0f}%")
    print(f"\nDetailed results:")
    for i, r in enumerate(results):
        status = '✅' if r['improvement'] > 50 else '⚠️' if r['improvement'] > 0 else '❌'
        print(f"  {status} Sample {i+1}: {r['improvement']:.0f}% improvement " +
              f"(deg={r['deg_mae']:.3f} → final={r['final_mae']:.3f})")
    
    print("\n" + "="*70)
    print("🎯 GENERALIZATION TEST RESULT")
    print("="*70)
    if success_rate > 70:
        print("✅ EXCELLENT generalization to external dataset!")
        print("   Model trained on VITON-HD works well on ATR data")
    elif success_rate > 50:
        print("⚠️  GOOD generalization with some challenges")
        print("   Model shows reasonable transfer to different data source")
    else:
        print("❌ Limited generalization to external data")
        print("   Model may be overfitting to VITON-HD characteristics")


if __name__ == '__main__':
    main()