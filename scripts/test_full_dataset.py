#!/usr/bin/env python3
"""
Comprehensive evaluation on entire test dataset
Generates detailed statistics, visualizations, and sample outputs
"""

import sys
import argparse
import yaml
import csv
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FastColorCorrectionUNet
from src.data import UpperMaskDegradedDataset
from src.evaluation import compute_metrics
from src.utils import extract_color_conditioning, visualize_results


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def evaluate_full_dataset(model, test_loader, device, img_size, output_dir, num_visualizations=10):
    """
    Run comprehensive evaluation on full test dataset
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: torch.device
        img_size: Target image size
        output_dir: Directory to save results
        num_visualizations: Number of example visualizations to create
    
    Returns:
        Dictionary with all metrics and statistics
    """
    model.eval()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Storage for all metrics
    all_metrics = defaultdict(list)
    sample_counter = 0
    
    print("="*70)
    print("FULL DATASET EVALUATION")
    print("="*70)
    print(f"\n📊 Testing on {len(test_loader.dataset)} samples")
    print(f"💾 Saving results to: {output_dir}")
    print(f"🎨 Creating {num_visualizations} example visualizations\n")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Prepare data
            degraded = batch['image'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            gt = batch['gt'].to(device, non_blocking=True)
            
            # Resize if needed
            if degraded.shape[-1] != img_size:
                degraded = F.interpolate(degraded, size=(img_size, img_size),
                                       mode='bilinear', align_corners=False)
                mask = F.interpolate(mask, size=(img_size, img_size), mode='nearest')
                gt = F.interpolate(gt, size=(img_size, img_size),
                                  mode='bilinear', align_corners=False)
            
            # Color conditioning
            mask_3ch = mask.expand(-1, 3, -1, -1)
            color = (gt * mask_3ch).sum(dim=(2,3)) / (mask_3ch.sum(dim=(2,3)) + 1e-6)
            color_cond = color[:, :, None, None].expand_as(gt)
            
            # Forward pass
            output = model(degraded, mask, color_cond)
            
            # Compute metrics for each sample in batch
            batch_size = degraded.shape[0]
            for i in range(batch_size):
                # Extract single sample
                pred_i = output[i:i+1]
                gt_i = gt[i:i+1]
                mask_i = mask[i:i+1]
                
                # Compute metrics
                metrics = compute_metrics(pred_i, gt_i, mask_i)
                
                # Store metrics
                for key, value in metrics.items():
                    all_metrics[key].append(value)
                
                # Create visualization for first N samples
                if sample_counter < num_visualizations:
                    # Get original (clean) image from gt
                    original_np = gt_i.squeeze(0).cpu().permute(1, 2, 0).numpy()
                    degraded_np = degraded[i:i+1].squeeze(0).cpu().permute(1, 2, 0).numpy()
                    corrected_np = pred_i.squeeze(0).cpu().permute(1, 2, 0).numpy()
                    mask_np = mask_i.squeeze().cpu().numpy()
                    
                    # Create visualization
                    viz_path = output_dir / f"sample_{sample_counter:04d}_visualization.png"
                    visualize_results(
                        torch.from_numpy(original_np).permute(2, 0, 1),
                        torch.from_numpy(degraded_np).permute(2, 0, 1),
                        torch.from_numpy(corrected_np).permute(2, 0, 1),
                        torch.from_numpy(mask_np),
                        save_path=viz_path
                    )
                    
                    sample_counter += 1
    
    # Compute statistics
    results = {}
    for key in all_metrics:
        values = np.array(all_metrics[key])
        results[f'{key}_mean'] = float(np.mean(values))
        results[f'{key}_std'] = float(np.std(values))
        results[f'{key}_median'] = float(np.median(values))
        results[f'{key}_min'] = float(np.min(values))
        results[f'{key}_max'] = float(np.max(values))
        results[f'{key}_values'] = values  # Keep for visualization
    
    return results, all_metrics


def plot_metric_distributions(all_metrics, save_path):
    """Create distribution plots for all metrics"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Metric Distributions Across Test Set', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('color_accuracy', 'Color Accuracy (MAE)', 'steelblue'),
        ('mse_global', 'MSE (Global)', 'coral'),
        ('mse_masked', 'MSE (Masked)', 'lightcoral'),
        ('psnr_global', 'PSNR (Global) [dB]', 'mediumseagreen'),
        ('psnr_masked', 'PSNR (Masked) [dB]', 'lightgreen'),
    ]
    
    for idx, (key, title, color) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = all_metrics[key]
        
        # Histogram
        ax.hist(values, bins=50, color=color, alpha=0.7, edgecolor='black')
        
        # Statistics lines
        mean_val = np.mean(values)
        median_val = np.median(values)
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2,
                   label=f'Median: {median_val:.4f}')
        
        ax.set_xlabel(title, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved metric distributions to: {save_path}")
    plt.close()


def plot_summary_statistics(results, save_path):
    """Create summary statistics visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics = ['color_accuracy', 'mse_global', 'mse_masked', 'psnr_global', 'psnr_masked']
    labels = ['Color Accuracy', 'MSE (Global)', 'MSE (Masked)', 'PSNR (Global)', 'PSNR (Masked)']
    
    means = [results[f'{m}_mean'] for m in metrics]
    stds = [results[f'{m}_std'] for m in metrics]
    
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7, 
                  color=['steelblue', 'coral', 'lightcoral', 'mediumseagreen', 'lightgreen'])
    
    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Summary Statistics (Mean ± Std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"📊 Saved summary statistics to: {save_path}")
    plt.close()


def save_results_csv(results, all_metrics, save_path):
    """Save detailed results to CSV"""
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['sample_idx', 'color_accuracy', 'mse_global', 'mse_masked', 
                        'psnr_global', 'psnr_masked'])
        
        # Data
        num_samples = len(all_metrics['color_accuracy'])
        for i in range(num_samples):
            row = [
                i,
                all_metrics['color_accuracy'][i],
                all_metrics['mse_global'][i],
                all_metrics['mse_masked'][i],
                all_metrics['psnr_global'][i],
                all_metrics['psnr_masked'][i]
            ]
            writer.writerow(row)
    
    print(f"💾 Saved detailed results to: {save_path}")


def print_results_table(results):
    """Print results in a nice table format"""
    print("\n" + "="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"{'Metric':<30} {'Mean':<15} {'Std':<15} {'Min':<15} {'Max':<15}")
    print("-"*70)
    
    metrics = ['color_accuracy', 'mse_global', 'mse_masked', 'psnr_global', 'psnr_masked']
    labels = {
        'color_accuracy': 'Color Accuracy (MAE)',
        'mse_global': 'MSE (Global)',
        'mse_masked': 'MSE (Masked)',
        'psnr_global': 'PSNR (Global) [dB]',
        'psnr_masked': 'PSNR (Masked) [dB]'
    }
    
    for metric in metrics:
        label = labels[metric]
        mean = results[f'{metric}_mean']
        std = results[f'{metric}_std']
        min_val = results[f'{metric}_min']
        max_val = results[f'{metric}_max']
        
        print(f"{label:<30} {mean:<15.6f} {std:<15.6f} {min_val:<15.6f} {max_val:<15.6f}")
    
    print("="*70)


def main(args):
    print("="*70)
    print("SHOOTIFY COLOR CORRECTION - FULL DATASET TESTING")
    print("="*70)
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        # Load from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        config = checkpoint['config']
    
    # Override test manifest if provided
    if args.test_manifest:
        config['data']['test_manifest'] = args.test_manifest
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Create test dataset
    print(f"\n📂 Loading test dataset...")
    test_dataset = UpperMaskDegradedDataset(
        config['data']['test_manifest'],
        img_size=config['evaluation']['img_size'],
        use_degradation=True,  # Apply degradation for testing
        degradation_strength=config['training'].get('degradation_strength', 0.5)
    )
    print(f"✅ Test: {len(test_dataset)} samples")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config['evaluation']['num_workers'],
        pin_memory=True
    )
    print(f"✅ Test batches: {len(test_loader)}")
    
    # Load model
    print(f"\n🔧 Loading model...")
    model = FastColorCorrectionUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded from: {args.checkpoint}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run evaluation
    print(f"\n🚀 Starting evaluation...\n")
    results, all_metrics = evaluate_full_dataset(
        model=model,
        test_loader=test_loader,
        device=device,
        img_size=config['evaluation']['img_size'],
        output_dir=output_dir,
        num_visualizations=args.num_visualizations
    )
    
    # Print results
    print_results_table(results)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    plot_metric_distributions(all_metrics, output_dir / 'metric_distributions.png')
    plot_summary_statistics(results, output_dir / 'summary_statistics.png')
    
    # Save detailed results
    print(f"\n💾 Saving detailed results...")
    save_results_csv(results, all_metrics, output_dir / 'detailed_results.csv')
    
    # Save summary
    summary_path = output_dir / 'summary.txt'
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SHOOTIFY COLOR CORRECTION - EVALUATION SUMMARY\n")
        f.write("="*70 + "\n\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test samples: {len(test_dataset)}\n")
        f.write(f"Device: {device}\n\n")
        f.write("="*70 + "\n")
        f.write("RESULTS\n")
        f.write("="*70 + "\n")
        f.write(f"{'Metric':<30} {'Mean':<15} {'Std':<15}\n")
        f.write("-"*70 + "\n")
        
        for metric in ['color_accuracy', 'mse_global', 'mse_masked', 'psnr_global', 'psnr_masked']:
            mean = results[f'{metric}_mean']
            std = results[f'{metric}_std']
            f.write(f"{metric:<30} {mean:<15.6f} {std:<15.6f}\n")
        
        f.write("="*70 + "\n")
    
    print(f"💾 Saved summary to: {summary_path}")
    
    print(f"\n{'='*70}")
    print("✅ EVALUATION COMPLETE")
    print(f"{'='*70}")
    print(f"\n📁 All results saved to: {output_dir}")
    print(f"   - metric_distributions.png")
    print(f"   - summary_statistics.png")
    print(f"   - detailed_results.csv")
    print(f"   - summary.txt")
    print(f"   - sample_XXXX_visualization.png (x{args.num_visualizations})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run comprehensive evaluation on full test dataset')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--test-manifest', type=str, help='Path to test manifest CSV (overrides config)')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for evaluation')
    parser.add_argument('--output-dir', type=str, default='outputs/full_test_results', 
                       help='Directory to save results')
    parser.add_argument('--num-visualizations', type=int, default=10, 
                       help='Number of example visualizations to create')
    
    args = parser.parse_args()
    main(args)
