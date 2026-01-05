#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import time
from torch.utils.data import DataLoader

from src.data import UpperMaskDegradedDataset
from src.models.fast_unet import MinimalUNet
from src.losses.simple_loss import simple_structure_loss


def test_minimal_model():
    print("="*70)
    print("TEST: MINIMAL U-NET SPEED")
    print("="*70)
    
    device = torch.device('cpu')
    
    # Minimal model
    print("\n  Creating MINIMAL model...")
    model = MinimalUNet(in_channels=7).to(device)
    model.train()
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    dataset = UpperMaskDegradedDataset(
        'data/train_manifest.csv',
        img_size=256,
        use_degradation=True,
        degradation_strength=0.5,
        cloth_mismatch_prob=0.3,
    )
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    batch = next(iter(dataloader))
    
    degraded = batch['image'].to(device)
    mask = batch['mask'].to(device)
    gt = batch['gt'].to(device)
    cloth = batch['cloth'].to(device)
    
    print(f"\n  Testing 5 iterations...")
    
    times = []
    for i in range(5):
        start = time.time()
        
        optimizer.zero_grad()
        output = model(degraded, mask, cloth)
        loss, components = simple_structure_loss(output, degraded, gt, mask)
        loss.backward()
        optimizer.step()
        
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"   Iteration {i+1}: {elapsed:.2f}s")
    
    avg_time = sum(times) / len(times)
    
    print(f"\n Results:")
    print(f"   Average: {avg_time:.2f}s")
    print(f"   Expected per epoch: {avg_time * 2912 / 60:.1f} minutes")
    print(f"   Total training (15 epochs): {avg_time * 2912 * 15 / 3600:.1f} hours")
    
    if avg_time < 5:
        print("\n EXCELLENT! Fast enough to train!")
    elif avg_time < 15:
        print("\n  Slower but workable")
    else:
        print("\n Still too slow - need different approach")


if __name__ == '__main__':
    test_minimal_model()