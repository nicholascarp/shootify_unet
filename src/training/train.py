#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 21:26:19 2025

@author: nicholascarp
"""

"""
Training logic for color correction model
"""

import time
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from tqdm.auto import tqdm

from .loss import ColorCorrectionLoss


class Trainer:
    """Trainer class for color correction model"""
    
    def __init__(self, model, train_loader, test_loader, config, device):
        """
        Args:
            model: FastColorCorrectionUNet model
            train_loader: Training data loader
            test_loader: Test data loader
            config: Configuration dict
            device: torch.device
        """
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config
        self.device = device
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Setup loss
        self.criterion = ColorCorrectionLoss(mask_weight=config['mask_weight'])
        
        # Mixed precision
        self.use_amp = config['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        
        # Gradient accumulation
        self.gradient_accumulation = config['gradient_accumulation']
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_global': [],
            'train_masked': [],
            'val_loss': [],
            'epoch_times': []
        }
        
        self.img_size = config['img_size']
    
    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_start = time.time()
        
        epoch_loss = 0
        epoch_global = 0
        epoch_masked = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", ncols=100, leave=True)
        
        for batch_idx, batch in enumerate(pbar):
            # Prepare data
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
            
            # Forward + Backward
            if self.use_amp:
                with autocast():
                    output = self.model(degraded, mask, color_cond)
                    loss, loss_global_val, loss_masked_val = self.criterion(output, gt, mask)
                    loss = loss / self.gradient_accumulation
                
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                output = self.model(degraded, mask, color_cond)
                loss, loss_global_val, loss_masked_val = self.criterion(output, gt, mask)
                loss = loss / self.gradient_accumulation
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Track metrics
            loss_val = loss.item() * self.gradient_accumulation
            epoch_loss += loss_val
            epoch_global += loss_global_val
            epoch_masked += loss_masked_val
            
            # Update progress bar
            if batch_idx % 10 == 0:
                pbar.set_postfix({'loss': f'{loss_val:.4f}'})
        
        pbar.close()
        
        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_loss = epoch_loss / len(self.train_loader)
        avg_global = epoch_global / len(self.train_loader)
        avg_masked = epoch_masked / len(self.train_loader)
        
        self.history['train_loss'].append(avg_loss)
        self.history['train_global'].append(avg_global)
        self.history['train_masked'].append(avg_masked)
        self.history['epoch_times'].append(epoch_time)
        
        return avg_loss, avg_global, avg_masked, epoch_time
    
    def validate(self):
        """Run validation"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Validation", leave=False):
                degraded = batch['image'].to(self.device, non_blocking=True)
                mask = batch['mask'].to(self.device, non_blocking=True)
                gt = batch['gt'].to(self.device, non_blocking=True)
                
                # Resize
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
                if self.use_amp:
                    with autocast():
                        output = self.model(degraded, mask, color_cond)
                        loss, _, _ = self.criterion(output, gt, mask)
                else:
                    output = self.model(degraded, mask, color_cond)
                    loss, _, _ = self.criterion(output, gt, mask)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(self.test_loader)
        self.history['val_loss'].append(avg_val_loss)
        
        return avg_val_loss
    
    def train(self, epochs):
        """Full training loop"""
        print("="*70)
        print("TRAINING START")
        print("="*70)
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*70}")
            print(f"EPOCH {epoch}/{epochs}")
            print(f"{'='*70}")
            
            # Train
            avg_loss, avg_global, avg_masked, epoch_time = self.train_epoch(epoch)
            
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"   Time: {epoch_time/60:.1f} minutes")
            print(f"   Avg loss: {avg_loss:.4f}")
            print(f"   Global: {avg_global:.4f}")
            print(f"   Masked: {avg_masked:.4f}")
            
            # Validate
            if epoch % 2 == 0:
                print(f"\n🔍 Running validation...")
                val_loss = self.validate()
                print(f"   Validation loss: {val_loss:.4f}")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        
        return self.history