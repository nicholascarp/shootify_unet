#!/usr/bin/env python3
"""
Main training script for color correction model
"""

import os
import sys
import random
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FastColorCorrectionUNet
from src.data import UpperMaskDegradedDataset
from src.training import Trainer
from src.utils import plot_training_history


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # Load configuration
    print("="*70)
    print("SHOOTIFY COLOR CORRECTION - TRAINING")
    print("="*70)
    
    config_path = args.config if args.config else Path(__file__).parent.parent / 'config' / 'config.yaml'
    config = load_config(config_path)
    
    # Override config with command line arguments
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    if args.train_manifest:
        config['data']['train_manifest'] = args.train_manifest
    if args.test_manifest:
        config['data']['test_manifest'] = args.test_manifest
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f"\n📱 Device: {device}")
    
    # Create datasets
    print(f"\n📂 Loading datasets...")
    train_dataset = UpperMaskDegradedDataset(
        config['data']['train_manifest'],
        img_size=config['training']['img_size'],
        use_degradation=True,
        degradation_strength=config['training']['degradation_strength']
    )
    
    test_dataset = UpperMaskDegradedDataset(
        config['data']['test_manifest'],
        img_size=config['training']['img_size'],
        use_degradation=False,
        degradation_strength=config['training']['degradation_strength']
    )
    
    print(f"✅ Train: {len(train_dataset)} samples")
    print(f"✅ Test: {len(test_dataset)} samples")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        persistent_workers=True if config['training']['num_workers'] > 0 else False,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
        shuffle=False,
        num_workers=config['evaluation']['num_workers'],
        pin_memory=config['training']['pin_memory'],
        persistent_workers=True if config['evaluation']['num_workers'] > 0 else False
    )
    
    print(f"✅ Train batches: {len(train_loader)}")
    print(f"✅ Test batches: {len(test_loader)}")
    
    # Create model
    print(f"\n🔧 Creating model...")
    model = FastColorCorrectionUNet(
        in_channels=config['model']['in_channels'],
        out_channels=config['model']['out_channels']
    )
    model = model.to(device)
    
    num_params = model.get_num_params()
    print(f"✅ Model created: {num_params:,} parameters")
    
    # Create trainer
    print(f"\n🏃 Setting up trainer...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config['training'],
        device=device
    )
    print(f"✅ Trainer ready")
    
    # Train
    print(f"\n🚀 Starting training for {config['training']['epochs']} epochs...")
    history = trainer.train(config['training']['epochs'])
    
    # Save model
    output_dir = Path(args.output_dir) if args.output_dir else Path('outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = output_dir / 'model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, model_path)
    print(f"\n💾 Model saved to: {model_path}")
    
    # Plot training history
    history_plot_path = output_dir / 'training_history.png'
    plot_training_history(history, save_path=history_plot_path)
    
    print(f"\n{'='*70}")
    print("✅ TRAINING COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train color correction model')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--train-manifest', type=str, help='Path to train manifest CSV')
    parser.add_argument('--test-manifest', type=str, help='Path to test manifest CSV')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    main(args)
