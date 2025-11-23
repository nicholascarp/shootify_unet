"""
Evaluation pipeline for color correction model
"""

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .metrics import compute_metrics


class Evaluator:
    """Evaluator class for color correction model"""
    
    def __init__(self, model, test_loader, device, img_size=256):
        """
        Args:
            model: FastColorCorrectionUNet model
            test_loader: Test data loader
            device: torch.device
            img_size: Target image size
        """
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.img_size = img_size
    
    def evaluate(self):
        """
        Run full evaluation on test set
        
        Returns:
            Dictionary with aggregated metrics
        """
        self.model.eval()
        
        all_metrics = {
            'color_accuracy': [],
            'mse_global': [],
            'mse_masked': [],
            'psnr_global': [],
            'psnr_masked': []
        }
        
        print("="*70)
        print("EVALUATION")
        print("="*70)
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Evaluating"):
                degraded = batch['image'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                gt = batch['gt'].to(self.device, non_blocking=True)
                
                # Resize if needed
                if degraded.shape[-1] != self.img_size:
                    degraded = F.interpolate(degraded, size=(self.img_size, self.img_size),
                                           mode='bilinear', align_corners=False)
                    mask = F.interpolate(mask, size=(self.img_size, self.img_size), mode='nearest')
                    gt = F.interpolate(gt, size=(self.img_size, self.img_size),
                                      mode='bilinear', align_corners=False)
                
                # Color conditioning
                mask_3ch = mask.expand(-1, 3, -1, -1)
                color = (gt * mask_3ch).sum(dim=(2,3)) / (mask_3ch.sum(dim=(2,3)) + 1e-6)
                color_cond = color[:, :, None, None].expand_as(gt)
                
                # Forward
                output = self.model(degraded, mask, color_cond)
                
                # Compute metrics
                metrics = compute_metrics(output, gt, mask)
                
                for key in all_metrics:
                    all_metrics[key].append(metrics[key])
        
        # Aggregate metrics
        results = {}
        for key in all_metrics:
            values = all_metrics[key]
            results[f'{key}_mean'] = sum(values) / len(values)
            results[f'{key}_std'] = (sum((x - results[f'{key}_mean'])**2 for x in values) / len(values))**0.5
        
        # Print results
        print("\n📊 Evaluation Results:")
        print("-" * 70)
        print(f"{'Metric':<30} {'Mean':<15} {'Std':<15}")
        print("-" * 70)
        
        for key in all_metrics:
            mean_key = f'{key}_mean'
            std_key = f'{key}_std'
            print(f"{key:<30} {results[mean_key]:<15.6f} {results[std_key]:<15.6f}")
        
        print("="*70)
        
        return results
