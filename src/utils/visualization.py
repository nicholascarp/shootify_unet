"""
Visualization utilities
"""
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualize_results(original, degraded, corrected, mask, save_path=None):
    """
    Visualize color correction results in a clear before/after flow
    
    Args:
        original: [3, H, W] tensor - Original clean image
        degraded: [3, H, W] tensor - Degraded input image
        corrected: [3, H, W] tensor - Corrected output image
        mask: [H, W] tensor or numpy array - Garment mask (2D)
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Convert to numpy and transpose to HWC
    original_np = original.cpu().permute(1, 2, 0).numpy()  # Changed from gt_np
    degraded_np = degraded.cpu().permute(1, 2, 0).numpy()
    corrected_np = corrected.cpu().permute(1, 2, 0).numpy()
    
    # Handle mask - ensure it's 2D
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    while mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    # Top row: Main progression
    axes[0, 0].imshow(original_np.clip(0, 1))  # Changed from gt_np
    axes[0, 0].set_title('1. Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(degraded_np.clip(0, 1))
    axes[0, 1].set_title('2. After Degradation', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(corrected_np.clip(0, 1))
    axes[0, 2].set_title('3. After Correction', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(mask_np, cmap='gray')
    axes[0, 3].set_title('Mask Used', fontsize=14, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Bottom row: Differences
    # Original vs Degraded
    deg_diff = np.abs(original_np - degraded_np) * 5  # Changed from gt_np
    axes[1, 0].imshow(deg_diff.clip(0, 1))
    axes[1, 0].set_title('Degradation Diff (5x)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Degraded vs Corrected
    correction = np.abs(degraded_np - corrected_np) * 5
    axes[1, 1].imshow(correction.clip(0, 1))
    axes[1, 1].set_title('Correction Applied (5x)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Corrected vs Original
    final_diff = np.abs(corrected_np - original_np) * 5  # Changed from gt_np
    axes[1, 2].imshow(final_diff.clip(0, 1))
    axes[1, 2].set_title('Remaining Error (5x)', fontsize=12)
    axes[1, 2].axis('off')
    
    # Masked region overlay
    masked_original = original_np.copy()  # Changed from gt_np
    masked_original[mask_np < 0.5] *= 0.3  # Darken non-masked areas
    axes[1, 3].imshow(masked_original.clip(0, 1))
    axes[1, 3].set_title('Masked Region', fontsize=12)
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    
    plt.close()

def plot_training_history(history, save_path=None):
    """
    Plot training history
    
    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    if history['val_loss']:
        val_epochs = [i*2 for i in range(1, len(history['val_loss']) + 1)]
        axes[0].plot(val_epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Component losses
    axes[1].plot(epochs, history['train_global'], 'g-', label='Global MSE')
    axes[1].plot(epochs, history['train_masked'], 'orange', label='Masked MSE')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss Components')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()
