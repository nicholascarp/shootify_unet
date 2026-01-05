#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-Quality Training Script for Color Correction
Implements Options 2+3+4:
- Option 2: Simple loss
- Option 3: Identity skip connection (in model)
- Option 4: Light Aware cloth (in dataset)
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.data import UpperMaskDegradedDataset
from src.models.fast_unet import MinimalUNet
from src.losses.simple_loss import simple_structure_loss


def get_device1():
    """Detect and return the best available device"""
    # Force CPU - MPS has severe performance issues with this architecture
    device = torch.device('cpu')
    device_type = 'cpu'
    print(f"💻 Using CPU (MPS has performance issues with this model)")
    print(f"   Expected: ~5-10s per batch on CPU vs 130s on MPS")
    
    return device, device_type

def get_device():
    """Detect and return the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        device_type = 'cuda'
        print(f" Using CUDA GPU")
    elif torch.backends.mps.is_available():
        # MPS can have issues, but let user decide
        device = torch.device('mps')
        device_type = 'mps'
        print(f" Using MPS (Apple Silicon)")
        print(f"   Note: MPS may be slower than CPU for this model")
    else:
        device = torch.device('cpu')
        device_type = 'cpu'
        print(f" Using CPU")
    
    return device, device_type


def train_epoch(model, dataloader, optimizer, device, device_type, epoch, scaler=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    running_components = {
        'recon': 0.0,
        'inside': 0.0,
    }
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        degraded = batch['image'].to(device)
        mask = batch['mask'].to(device)
        gt = batch['gt'].to(device)
        cloth = batch['cloth'].to(device)
        
        optimizer.zero_grad()
        
        if device_type == 'cuda' and scaler is not None:
            with torch.amp.autocast('cuda'):
                output = model(degraded, mask, cloth)
                # Use SIMPLE loss
                loss, components = simple_structure_loss(output, degraded, gt, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(degraded, mask, cloth)
            # Use SIMPLE loss
            loss, components = simple_structure_loss(output, degraded, gt, mask)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item()
        for key, value in components.items():
            running_components[key] += value
        
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.6f}'})
    
    avg_loss = running_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in running_components.items()}
    
    return avg_loss, avg_components


@torch.no_grad()
def validate(model, dataloader, device, device_type):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    running_components = {
        'recon': 0.0,
        'inside': 0.0,
    }
    
    pbar = tqdm(dataloader, desc="Validating")
    
    for batch in pbar:
        degraded = batch['image'].to(device)
        mask = batch['mask'].to(device)
        gt = batch['gt'].to(device)
        cloth = batch['cloth'].to(device)
        
        if device_type == 'cuda':
            with torch.amp.autocast('cuda'):
                output = model(degraded, mask, cloth)
                # Use SIMPLE loss
                loss, components = simple_structure_loss(output, degraded, gt, mask)
        else:
            output = model(degraded, mask, cloth)
            # Use SIMPLE loss
            loss, components = simple_structure_loss(output, degraded, gt, mask)
        
        running_loss += loss.item()
        for key, value in components.items():
            running_components[key] += value
    
    avg_loss = running_loss / len(dataloader)
    avg_components = {k: v / len(dataloader) for k, v in running_components.items()}
    
    return avg_loss, avg_components

def main():
    parser = argparse.ArgumentParser(description='Train production-quality color correction model')
    parser.add_argument('--train-manifest', type=str, required=True,
                       help='Path to training manifest CSV')
    parser.add_argument('--test-manifest', type=str, required=True,
                       help='Path to test manifest CSV')
    parser.add_argument('--output-dir', type=str, default='outputs/cloth_model_production',
                       help='Output directory for models and logs')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--num-workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--img-size', type=int, default=256,
                       help='Image size')
    parser.add_argument('--degradation-strength', type=float, default=0.5,
                       help='Degradation strength')
    parser.add_argument('--cloth-mismatch-prob', type=float, default=0.3,
                       help='Probability of using mismatched cloth')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("PRODUCTION-QUALITY COLOR CORRECTION TRAINING")
    print("="*70)
    print("\n Configuration:")
    print(f"   Train manifest: {args.train_manifest}")
    print(f"   Test manifest: {args.test_manifest}")
    print(f"   Output dir: {args.output_dir}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Image size: {args.img_size}")
    print(f"   Degradation strength: {args.degradation_strength}")
    print(f"   Cloth mismatch prob: {args.cloth_mismatch_prob}")
    
    # Setup device
    device, device_type = get_device()
    
    # Create datasets
    print("\n Loading datasets...")
    train_dataset = UpperMaskDegradedDataset(
        args.train_manifest,
        img_size=args.img_size,
        use_degradation=True,
        degradation_strength=args.degradation_strength,
        cloth_mismatch_prob=args.cloth_mismatch_prob,
    )
    
    test_dataset = UpperMaskDegradedDataset(
        args.test_manifest,
        img_size=args.img_size,
        use_degradation=True,
        degradation_strength=args.degradation_strength,
        cloth_mismatch_prob=args.cloth_mismatch_prob,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device_type == 'cuda')
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device_type == 'cuda')
    )
    
    print(f" Train samples: {len(train_dataset)} ({len(train_loader)} batches)")
    print(f" Test samples: {len(test_dataset)} ({len(test_loader)} batches)")
    
    # Create model
    print("\n🏗️  Creating model...")
    model = MinimalUNet(in_channels=7)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f" Model created with {num_params:,} parameters")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Gradient scaler for mixed precision (CUDA only)
    scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_components': [],
        'val_components': [],
    }
    
    best_val_loss = float('inf')
    
    # Training loop
    print("\n Starting training...")
    print("="*70)
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*70)
        
        # Train
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, device, device_type, epoch, scaler
        )
        
        # Validate
        val_loss, val_components = validate(model, test_loader, device, device_type)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_components'].append(train_components)
        history['val_components'].append(val_components)
        
        # Print results
        print(f"\n Epoch {epoch} Results:")
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss:   {val_loss:.6f}")
        print(f"\n   Train Components:")
        for key, value in train_components.items():
            print(f"      {key:12s}: {value:.6f}")
        print(f"\n   Val Components:")
        for key, value in val_components.items():
            print(f"      {key:12s}: {value:.6f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        
        checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.output_dir, 'model_best.pth')
            torch.save(checkpoint, best_path)
            print(f" Saved best model: {best_path}")
        else:
            print(f" Saved checkpoint: {checkpoint_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n Saved training history: {history_path}")
    
    print("\n" + "="*70)
    print(" TRAINING COMPLETE!")
    print("="*70)
    print(f"\n Best validation loss: {best_val_loss:.6f}")
    print(f" Output directory: {args.output_dir}")
    print(f" Best model: {os.path.join(args.output_dir, 'model_best.pth')}")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()