"""
Training package
"""

from .loss import ColorCorrectionLoss
from .train import Trainer

__all__ = ['ColorCorrectionLoss', 'Trainer']
