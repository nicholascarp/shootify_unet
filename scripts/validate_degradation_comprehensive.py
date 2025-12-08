#!/usr/bin/env python3
"""
Comprehensive degradation validation across entire dataset
Tests that degradation works reliably on ALL color ranges
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from src.data import UpperMaskDegradedDataset


def validate_degradation_comprehensive(num_samples=1000):
    """
    Test degradation on large sample of dataset
    
    Args:
        num_samples: Number of samples to test (default 1000)
    """
    print("="*70)
    print("COMPREHENSIVE DEGRADATION VALIDATION")
    print("="*70)
    
    dataset = UpperMaskDegradedDataset(
        'data/train_manifest.csv',
        img_size=256,
        use_degradation=True,
        degradation_strength=0.5,
        cloth_mismatch_prob=0.0,
    )
    
    num_samples = min(num_samples, len(dataset))
    print(f"\n📊 Testing {num_samples} samples from {len(dataset)} total...")
    
    # Statistics
    stats = {
        'dark': [],      # brightness < 0.2
        'medium': [],    # 0.2 <= brightness < 0.5
        'bright': [],    # brightness >= 0.5
    }
    
    failed_samples = []  # Samples where degradation failed
    degradation_values = []
    
    # Test samples
    print("\n🔍 Testing degradation effectiveness...")
    for i in tqdm(range(num_samples)):
        try:
            sample = dataset[i]
            
            gt = sample['gt']
            degraded = sample['image']
            mask = sample['mask']
            
            # Calculate degradation in masked region
            mask_bool = mask.squeeze() > 0.5
            
            if mask_bool.sum() < 100:
                continue  # Skip if mask too small
            
            # Mean brightness of original
            mean_color = gt[:, mask_bool].mean(dim=1)
            brightness = mean_color.mean().item()
            
            # Degradation amount (MAE in masked region)
            degradation_mae = (degraded[:, mask_bool] - gt[:, mask_bool]).abs().mean().item()
            
            # Categorize by brightness
            if brightness < 0.2:
                category = 'dark'
            elif brightness < 0.5:
                category = 'medium'
            else:
                category = 'bright'
            
            stats[category].append(degradation_mae)
            degradation_values.append(degradation_mae)
            
            # Flag if degradation too small
            if degradation_mae < 0.05:
                failed_samples.append({
                    'index': i,
                    'category': category,
                    'brightness': brightness,
                    'degradation_mae': degradation_mae
                })
        
        except Exception as e:
            print(f"\n❌ Error on sample {i}: {e}")
            continue
    
    # Print results
    print("\n" + "="*70)
    print("📊 VALIDATION RESULTS")
    print("="*70)
    
    print(f"\nTotal samples tested: {num_samples}")
    print(f"Valid samples: {len(degradation_values)}")
    
    print("\n📈 Degradation Statistics by Color Range:")
    print("-"*70)
    
    for category in ['dark', 'medium', 'bright']:
        values = stats[category]
        if len(values) > 0:
            mean_deg = np.mean(values)
            std_deg = np.std(values)
            min_deg = np.min(values)
            max_deg = np.max(values)
            failed_count = sum(1 for v in values if v < 0.05)
            
            print(f"\n{category.upper()} colors (n={len(values)}):")
            print(f"  Mean degradation: {mean_deg:.4f}")
            print(f"  Std deviation:    {std_deg:.4f}")
            print(f"  Range:            [{min_deg:.4f}, {max_deg:.4f}]")
            print(f"  Failed (<0.05):   {failed_count} ({failed_count/len(values)*100:.1f}%)")
            
            if mean_deg < 0.1:
                print(f"  ⚠️  WARNING: Low degradation for {category} colors!")
    
    # Overall statistics
    print("\n📊 Overall Statistics:")
    print("-"*70)
    mean_overall = np.mean(degradation_values)
    std_overall = np.std(degradation_values)
    min_overall = np.min(degradation_values)
    max_overall = np.max(degradation_values)
    
    print(f"Mean degradation (all):  {mean_overall:.4f}")
    print(f"Std deviation:           {std_overall:.4f}")
    print(f"Range:                   [{min_overall:.4f}, {max_overall:.4f}]")
    
    # Failure analysis
    print("\n❌ Failed Samples (degradation < 0.05):")
    print("-"*70)
    if len(failed_samples) == 0:
        print("✅ NO FAILURES! All samples degraded successfully!")
    else:
        print(f"Total failures: {len(failed_samples)} ({len(failed_samples)/len(degradation_values)*100:.1f}%)")
        print("\nTop 10 worst cases:")
        failed_samples.sort(key=lambda x: x['degradation_mae'])
        for i, sample in enumerate(failed_samples[:10]):
            print(f"  {i+1}. Index {sample['index']:5d} | "
                  f"{sample['category']:7s} | "
                  f"brightness={sample['brightness']:.3f} | "
                  f"degradation={sample['degradation_mae']:.4f}")
    
    # Visualize distribution
    print("\n📊 Creating distribution plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Overall histogram
    axes[0, 0].hist(degradation_values, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(0.05, color='red', linestyle='--', linewidth=2, label='Failure threshold')
    axes[0, 0].set_xlabel('Degradation MAE', fontsize=11)
    axes[0, 0].set_ylabel('Count', fontsize=11)
    axes[0, 0].set_title(f'Degradation Distribution\n(n={len(degradation_values)}, mean={mean_overall:.4f})', 
                         fontsize=12, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. By category
    categories = ['dark', 'medium', 'bright']
    colors = ['darkblue', 'orange', 'yellow']
    positions = []
    data_to_plot = []
    
    for cat in categories:
        if len(stats[cat]) > 0:
            data_to_plot.append(stats[cat])
            positions.append(cat)
    
    if len(data_to_plot) > 0:
        bp = axes[0, 1].boxplot(data_to_plot, labels=positions, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(data_to_plot)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[0, 1].axhline(0.05, color='red', linestyle='--', linewidth=2, label='Failure threshold')
        axes[0, 1].set_ylabel('Degradation MAE', fontsize=11)
        axes[0, 1].set_title('Degradation by Color Range', fontsize=12, fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative distribution
    sorted_values = np.sort(degradation_values)
    cumulative = np.arange(1, len(sorted_values) + 1) / len(sorted_values) * 100
    
    axes[1, 0].plot(sorted_values, cumulative, linewidth=2)
    axes[1, 0].axvline(0.05, color='red', linestyle='--', linewidth=2, label='Failure threshold')
    axes[1, 0].axhline(95, color='green', linestyle=':', linewidth=1.5, label='95% threshold')
    axes[1, 0].set_xlabel('Degradation MAE', fontsize=11)
    axes[1, 0].set_ylabel('Cumulative %', fontsize=11)
    axes[1, 0].set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Statistics table
    axes[1, 1].axis('off')
    
    table_data = []
    table_data.append(['Category', 'Count', 'Mean', 'Std', 'Min', 'Failures'])
    
    for cat in categories:
        if len(stats[cat]) > 0:
            values = stats[cat]
            failures = sum(1 for v in values if v < 0.05)
            table_data.append([
                cat.upper(),
                f"{len(values)}",
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{failures} ({failures/len(values)*100:.1f}%)"
            ])
    
    table_data.append(['─'*10]*6)
    table_data.append([
        'OVERALL',
        f"{len(degradation_values)}",
        f"{mean_overall:.4f}",
        f"{std_overall:.4f}",
        f"{min_overall:.4f}",
        f"{len(failed_samples)} ({len(failed_samples)/len(degradation_values)*100:.1f}%)"
    ])
    
    table = axes[1, 1].table(cellText=table_data, cellLoc='center', loc='center',
                             colWidths=[0.15, 0.12, 0.15, 0.15, 0.15, 0.28])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(6):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Style total row
    for i in range(6):
        table[(len(table_data)-1, i)].set_facecolor('#E3F2FD')
        table[(len(table_data)-1, i)].set_text_props(weight='bold')
    
    plt.suptitle('Comprehensive Degradation Validation Report', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/degradation_validation_comprehensive.png', dpi=150, bbox_inches='tight')
    print(f"✅ Saved: outputs/degradation_validation_comprehensive.png")
    plt.show()
    
    # Final verdict
    print("\n" + "="*70)
    print("🎯 FINAL VERDICT")
    print("="*70)
    
    failure_rate = len(failed_samples) / len(degradation_values) * 100
    
    if failure_rate < 1.0:
        print("✅ EXCELLENT! Degradation works reliably (<1% failures)")
        print("   ✅ Safe to proceed with training!")
    elif failure_rate < 5.0:
        print("⚠️  ACCEPTABLE! Some degradation failures (1-5%)")
        print("   ✅ Should be OK for training, but monitor results")
    else:
        print("❌ WARNING! High degradation failure rate (>5%)")
        print("   ⚠️  Review failed samples before training!")
    
    print("\n💡 Recommendations:")
    if mean_overall > 0.15:
        print("   ✅ Mean degradation is good (>0.15)")
    else:
        print("   ⚠️  Consider increasing degradation strength")
    
    if min_overall > 0.05:
        print("   ✅ All samples have visible degradation")
    else:
        print("   ⚠️  Some samples have minimal degradation")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to test (default: 1000)')
    args = parser.parse_args()
    
    validate_degradation_comprehensive(args.num_samples)