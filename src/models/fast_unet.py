#!/usr/bin/env python3
"""
Minimal U-Net - 50% fewer layers for 10x faster training
"""
import torch
import torch.nn as nn


class MinimalUNet(nn.Module):
    """
    Minimal U-Net with only 2 encoder/decoder levels
    Much faster than full U-Net
    """
    
    def __init__(self, in_channels=7):
        super().__init__()
        
        # Encoder (only 2 levels instead of 4)
        self.enc1 = self._block(in_channels, 32)      # 7 -> 32
        self.enc2 = self._block(32, 64)               # 32 -> 64
        
        self.pool = nn.MaxPool2d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._block(64, 128)        # 64 -> 128
        
        # Decoder (only 2 levels)
        self.upconv2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = self._block(128, 64)              # 128 (concat) -> 64
        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = self._block(64, 32)               # 64 (concat) -> 32
        
        # Output
        self.out = nn.Conv2d(32, 3, 1)
    
    def _block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, degraded, mask, cloth):
        """
        Forward with identity skip
        """
        x = torch.cat([degraded, mask, cloth], dim=1)  # [B, 7, H, W]
        
        # Encoder
        e1 = self.enc1(x)           # [B, 32, H, W]
        e2 = self.enc2(self.pool(e1))  # [B, 64, H/2, W/2]
        
        # Bottleneck
        b = self.bottleneck(self.pool(e2))  # [B, 128, H/4, W/4]
        
        # Decoder
        d2 = self.upconv2(b)           # [B, 64, H/2, W/2]
        d2 = self.dec2(torch.cat([d2, e2], dim=1))  # [B, 64, H/2, W/2]
        
        d1 = self.upconv1(d2)          # [B, 32, H, W]
        d1 = self.dec1(torch.cat([d1, e1], dim=1))  # [B, 32, H, W]
        
        # Correction
        correction = self.out(d1)      # [B, 3, H, W]
        
        # Identity skip - only correct inside mask
        mask_3ch = mask.expand(-1, 3, -1, -1)
        output = degraded + correction * mask_3ch
        
        return output