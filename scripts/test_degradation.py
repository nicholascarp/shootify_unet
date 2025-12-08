#!/usr/bin/env python3
"""
Test lighting-aware cloth generation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data import UpperMaskDegradedDataset


def test_lighting_aware():
    print("="*70)
    print("TEST: LIGHTING-AWARE CLOTH GENERATION")
    print("="*70)
    
    dataset = UpperMaskDegradedDataset(
        'data/train_manifest.csv',
        img_size=256,
        use_degradation=True,
        degradation_strength=0.5,
        cloth_mismatch_prob=0.0,  # No mismatch for testing
    )
    
    print(f"\n📊 Sampling 6 items...")
    
    fig, axes = plt.subplots(6, 5, figsize=(20, 24))
    
    for i in range(6):
        sample = dataset[i]
        
        degraded = sample['image']
        cloth = sample['cloth']
        mask = sample['mask']
        gt = sample['gt']
        
        def to_img(t):
            return t.permute(1, 2, 0).numpy().clip(0, 1)
        
        # Ground truth
        axes[i, 0].imshow(to_img(gt))
        axes[i, 0].set_title(f'Sample {i+1}\nOriginal', fontsize=10)
        axes[i, 0].axis('off')
        
        # Lighting-aware cloth
        axes[i, 1].imshow(to_img(cloth))
        axes[i, 1].set_title('Lighting-Aware Cloth\n(Color+Lighting)', fontsize=10)
        axes[i, 1].axis('off')
        
        # Show luminance pattern
        lum = 0.299*gt[0] + 0.587*gt[1] + 0.114*gt[2]
        axes[i, 2].imshow(lum.numpy(), cmap='gray')
        axes[i, 2].set_title('Original Luminance\n(Lighting Pattern)', fontsize=10)
        axes[i, 2].axis('off')
        
        # Mask
        axes[i, 3].imshow(mask.squeeze().numpy(), cmap='gray')
        axes[i, 3].set_title(f'Mask\n({mask.mean()*100:.1f}%)', fontsize=10)
        axes[i, 3].axis('off')
        
        # Degraded
        axes[i, 4].imshow(to_img(degraded))
        axes[i, 4].set_title('Degraded Input', fontsize=10)
        axes[i, 4].axis('off')
    
    plt.suptitle('Lighting-Aware Cloth Test\n' +
                 'Cloth should have same lighting pattern (shadows/highlights) as original',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/test_lighting_aware_cloth.png', dpi=150)
    print(f"✅ Saved: outputs/test_lighting_aware_cloth.png")
    plt.show()
    
    print("\n✅ TEST COMPLETE")
    print("\nCheck that:")
    print("  1. Lighting-aware cloth has color fields (not uniform)")
    print("  2. Dark regions in original → dark in cloth")
    print("  3. Bright regions in original → bright in cloth")
    print("  4. Cloth preserves fold/shadow patterns")


if __name__ == '__main__':
    test_lighting_aware()