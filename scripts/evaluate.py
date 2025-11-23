#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:27:48 2025

@author: nicholascarp
"""

#!/usr/bin/env python3
"""
Evaluation script for color correction model
"""

import sys
import argparse
import yaml
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import FastColorCorrectionUNet
from src.data import UpperMaskDegradedDataset
from src.evaluation import Evaluator


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    print("="*70)
    print("SHOOTIFY COLOR CORRECTION - EVALUATION")
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
        use_degradation=False
    )
    print(f"✅ Test: {len(test_dataset)} samples")
    
    # Create dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['evaluation']['batch_size'],
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
    
    # Create evaluator
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        device=device,
        img_size=config['evaluation']['img_size']
    )
    
    # Run evaluation
    results = evaluator.evaluate()
    
    print(f"\n✅ EVALUATION COMPLETE")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate color correction model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, help='Path to config file (optional, will use config from checkpoint)')
    parser.add_argument('--test-manifest', type=str, help='Path to test manifest CSV (overrides config)')
    
    args = parser.parse_args()
    main(args)
