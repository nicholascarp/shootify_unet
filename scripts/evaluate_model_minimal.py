#!/usr/bin/env python3
"""
Evaluation script for MinimalUNet production model
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data import UpperMaskDegradedDataset
from src.models.fast_unet import MinimalUNet
from src.losses.simple_loss import simple_structure_loss


def get_device():
    """Get device - force CPU for stability"""
    return torch.device('cpu'), 'cpu'


def compute_metrics(output, gt, mask):
    """Compute evaluation metrics"""
    # Global metrics
    mse_global = torch.mean((output - gt) ** 2).item()
    psnr_global = 10 * np.log10(1.0 / (mse_global + 1e-10))
    
    # Masked metrics
    mask_3ch = mask.expand(-1, 3, -1, -1)
    output_masked = output[mask_3ch > 0.5]
    gt_masked = gt[mask_3ch > 0.5]
    
    if len(output_masked) > 0:
        mse_masked = torch.mean((output_masked - gt_masked) ** 2).item()
        psnr_masked = 10 * np.log10(1.0 / (mse_masked + 1e-10))
        color_mae = torch.mean(torch.abs(output_masked - gt_masked)).item()
    else:
        mse_masked = 0.0
        psnr_masked = 0.0
        color_mae = 0.0
    
    return {
        'mse_global': mse_global,
        'psnr_global': psnr_global,
        'mse_masked': mse_masked,
        'psnr_masked': psnr_masked,
        'color_mae': color_mae,
    }


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    
    all_outputs = []
    all_gts = []
    all_degraded = []
    all_masks = []
    all_cloths = []
    
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc="Evaluating")
    
    for batch in pbar:
        degraded = batch['image'].to(device)
        mask = batch['mask'].to(device)
        gt = batch['gt'].to(device)
        cloth = batch['cloth'].to(device)
        
        output = model(degraded, mask, cloth)
        
        loss, _ = simple_structure_loss(output, degraded, gt, mask)
        running_loss += loss.item()
        
        all_outputs.append(output.cpu())
        all_gts.append(gt.cpu())
        all_degraded.append(degraded.cpu())
        all_masks.append(mask.cpu())
        all_cloths.append(cloth.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    all_degraded = torch.cat(all_degraded, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_cloths = torch.cat(all_cloths, dim=0)
    
    avg_loss = running_loss / len(dataloader)
    
    return all_outputs, all_gts, all_degraded, all_masks, all_cloths, avg_loss


def visualize_samples(outputs, gts, degraded, masks, cloths, num_vis, save_path):
    """Create visualization grid"""
    fig, axes = plt.subplots(num_vis, 6, figsize=(24, 4*num_vis))
    
    def to_img(t):
        return t.permute(1, 2, 0).numpy().clip(0, 1)
    
    for i in range(num_vis):
        # Ground truth
        axes[i, 0].imshow(to_img(gts[i]))
        axes[i, 0].set_title(f'Sample {i+1}\nGround Truth', fontsize=10)
        axes[i, 0].axis('off')
        
        # Cloth (lighting-aware)
        axes[i, 1].imshow(to_img(cloths[i]))
        axes[i, 1].set_title('Cloth Reference\n(Lighting-Aware)', fontsize=10, color='green')
        axes[i, 1].axis('off')
        
        # Degraded
        axes[i, 2].imshow(to_img(degraded[i]))
        axes[i, 2].set_title('Degraded Input', fontsize=10)
        axes[i, 2].axis('off')
        
        # Mask
        axes[i, 3].imshow(masks[i].squeeze().numpy(), cmap='gray')
        axes[i, 3].set_title(f'Mask\n({masks[i].mean()*100:.1f}%)', fontsize=10)
        axes[i, 3].axis('off')
        
        # Output
        axes[i, 4].imshow(to_img(outputs[i]))
        axes[i, 4].set_title('Model Output', fontsize=10)
        axes[i, 4].axis('off')
        
        # Error
        error = (outputs[i] - gts[i]).abs() * 5
        axes[i, 5].imshow(to_img(error))
        mae = (outputs[i] - gts[i]).abs().mean().item()
        axes[i, 5].set_title(f'Error (5x)\nMAE: {mae:.4f}', fontsize=10)
        axes[i, 5].axis('off')
    
    plt.suptitle('V2.2 FINAL EVALUATION - Lighting-Aware Model', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"✅ Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--test-manifest', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-vis', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='outputs/evaluation_v2_2_final')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("V2.2 FINAL MODEL EVALUATION")
    print("Lighting-Aware Cloth + Guaranteed Degradation")
    print("="*70)
    
    device, device_type = get_device()
    print(f"\n🎮 Device: {device_type.upper()}")
    
    # Load dataset with REALISTIC settings (0% cloth mismatch)
    print(f"\n📂 Loading test dataset...")
    test_dataset = UpperMaskDegradedDataset(
        args.test_manifest,
        img_size=256,
        use_degradation=True,
        degradation_strength=0.5,
        cloth_mismatch_prob=0.0,  # ← REALISTIC: 0% mismatch!
        cloth_downsample_size=32,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✅ Loaded {len(test_dataset)} test samples ({len(test_loader)} batches)")
    print(f"   Cloth mismatch: 0% (realistic production scenario)")
    
    # Load model
    print(f"\n🏗️  Loading model...")
    model = MinimalUNet(in_channels=7)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    epoch = checkpoint.get('epoch', '?')
    train_loss = checkpoint.get('train_loss', '?')
    val_loss = checkpoint.get('val_loss', '?')
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded:")
    print(f"   Parameters: {num_params:,}")
    print(f"   Trained epoch: {epoch}")
    print(f"   Train loss: {train_loss}")
    print(f"   Val loss: {val_loss}")
    
    # Evaluate
    print(f"\n🔍 Evaluating model...")
    outputs, gts, degraded, masks, cloths, avg_loss = evaluate(model, test_loader, device)
    
    print(f"✅ Evaluation complete")
    print(f"   Average loss: {avg_loss:.6f}")
    
    # Compute metrics
    print(f"\n📊 Computing metrics...")
    metrics = compute_metrics(outputs, gts, masks)
    
    print(f"\n{'='*70}")
    print("📊 FINAL EVALUATION RESULTS - V2.2 LIGHTING-AWARE MODEL")
    print("="*70)
    print(f"\nGlobal Metrics (Whole Image):")
    print(f"  MSE:  {metrics['mse_global']:.6f}")
    print(f"  PSNR: {metrics['psnr_global']:.2f} dB")
    print(f"\nMasked Metrics (Garment Only):")
    print(f"  MSE:  {metrics['mse_masked']:.6f}")
    print(f"  PSNR: {metrics['psnr_masked']:.2f} dB")
    print(f"  Color MAE: {metrics['color_mae']:.6f}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\n💾 Saved metrics: {metrics_path}")
    
    # Visualizations
    print(f"\n📊 Creating visualizations...")
    vis_path = os.path.join(args.output_dir, 'evaluation_samples.png')
    visualize_samples(outputs, gts, degraded, masks, cloths, args.num_vis, vis_path)
    
    # Comparison with previous models
    print(f"\n{'='*70}")
    print("📈 COMPARISON WITH PREVIOUS MODELS")
    print("="*70)
    print(f"\n{'Model':<20} {'Color MAE':<12} {'PSNR (dB)':<12} {'Status'}")
    print("-"*70)
    print(f"{'V2':<20} {'0.084':<12} {'17.14':<12} {'Baseline'}")
    print(f"{'V2.1':<20} {'0.068':<12} {'18.81':<12} {'Better'}")
    print(f"{'V2.2 Lighting':<20} {metrics['color_mae']:<12.6f} {metrics['psnr_masked']:<12.2f} {'FINAL'}")
    
    # Calculate improvement
    improvement_vs_v21 = ((0.068 - metrics['color_mae']) / 0.068) * 100
    improvement_vs_v2 = ((0.084 - metrics['color_mae']) / 0.084) * 100
    
    print(f"\n🎯 Improvement:")
    print(f"   vs V2.1 (Original): {improvement_vs_v21:+.1f}%")
    print(f"   vs V2 (Smooth):     {improvement_vs_v2:+.1f}%")
    
    print("\n" + "="*70)
    print("✅ EVALUATION COMPLETE!")
    print("="*70)


if __name__ == '__main__':
    main()