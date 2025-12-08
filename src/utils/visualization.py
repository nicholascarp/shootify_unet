"""
Visualization utilities - IMPROVED VERSION
"""
import numpy as np
import matplotlib.pyplot as plt
import torch


def visualize_results(original, degraded, corrected, mask, save_path=None, show_metrics=True):
    """
    Visualize color correction results in a clear before/after flow
    
    Args:
        original: [3, H, W] tensor - Original clean image
        degraded: [3, H, W] tensor - Degraded input image
        corrected: [3, H, W] tensor - Corrected output image
        mask: [H, W] tensor or numpy array - Garment mask (2D)
        save_path: Optional path to save figure
        show_metrics: If True, display Color MAE and success status
    """
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Convert to numpy and transpose to HWC
    original_np = original.cpu().permute(1, 2, 0).numpy()
    degraded_np = degraded.cpu().permute(1, 2, 0).numpy()
    corrected_np = corrected.cpu().permute(1, 2, 0).numpy()
    
    # Handle mask - ensure it's 2D
    if torch.is_tensor(mask):
        mask_np = mask.cpu().numpy()
    else:
        mask_np = mask
    while mask_np.ndim > 2:
        mask_np = mask_np.squeeze()
    
    # Calculate Color MAE in masked region (if requested)
    if show_metrics:
        mask_3ch = np.stack([mask_np] * 3, axis=-1)
        masked_original = original_np[mask_np > 0.5]
        masked_corrected = corrected_np[mask_np > 0.5]
        if len(masked_original) > 0:
            color_mae = np.abs(masked_original - masked_corrected).mean()
            success_str = "✓ SUCCESS" if color_mae < 0.05 else "✗ NEEDS IMPROVEMENT"
            fig.suptitle(f'Color MAE: {color_mae:.4f} {success_str}', 
                        fontsize=16, fontweight='bold', y=0.98)
    
    # Top row: Main progression
    axes[0, 0].imshow(original_np.clip(0, 1))
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
    deg_diff = np.abs(original_np - degraded_np) * 5
    axes[1, 0].imshow(deg_diff.clip(0, 1))
    axes[1, 0].set_title('Degradation Diff (5x)', fontsize=12)
    axes[1, 0].axis('off')
    
    # Degraded vs Corrected
    correction = np.abs(degraded_np - corrected_np) * 5
    axes[1, 1].imshow(correction.clip(0, 1))
    axes[1, 1].set_title('Correction Applied (5x)', fontsize=12)
    axes[1, 1].axis('off')
    
    # Corrected vs Original
    final_diff = np.abs(corrected_np - original_np) * 5
    axes[1, 2].imshow(final_diff.clip(0, 1))
    axes[1, 2].set_title('Remaining Error (5x)', fontsize=12)
    axes[1, 2].axis('off')
    
    # Masked region overlay
    masked_original = original_np.copy()
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
    Plot training history - FIXED VERSION
    
    Args:
        history: Dictionary with training metrics. Expected keys:
            - 'train_loss': List of training losses
            - 'val_loss': List of validation losses (optional)
            - 'val_epochs': List of validation epoch numbers (optional)
            - 'train_global': List of global loss components (optional)
            - 'train_masked': List of masked loss components (optional)
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    train_epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(train_epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    
    # Validation loss (FIXED: flexible epoch handling)
    if history.get('val_loss'):
        if 'val_epochs' in history:
            # Use provided validation epoch numbers
            val_epochs = history['val_epochs']
        else:
            # Assume validation happens at same epochs as training
            val_epochs = range(1, len(history['val_loss']) + 1)
        
        axes[0].plot(val_epochs, history['val_loss'], 'r-', label='Val Loss', 
                    linewidth=2, marker='o', markersize=4)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Component losses (if available)
    if 'train_global' in history and 'train_masked' in history:
        axes[1].plot(train_epochs, history['train_global'], 'g-', 
                    label='Global MSE', linewidth=2)
        axes[1].plot(train_epochs, history['train_masked'], 'orange', 
                    label='Masked MSE', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Loss Components', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
    else:
        # If component losses not available, hide second plot
        axes[1].text(0.5, 0.5, 'Component losses not available', 
                    ha='center', va='center', fontsize=14, transform=axes[1].transAxes)
        axes[1].set_xticks([])
        axes[1].set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_batch_results(batch_images, save_path=None, grid_size=(2, 4)):
    """
    Plot multiple correction results in a grid
    
    Args:
        batch_images: List of tuples (original, degraded, corrected, mask)
        save_path: Optional path to save figure
        grid_size: Tuple (rows, cols) for grid layout
    """
    n_rows, n_cols = grid_size
    n_samples = min(len(batch_images), n_rows * n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols * 3, figsize=(n_cols * 9, n_rows * 3))
    
    for idx in range(n_samples):
        original, degraded, corrected, mask = batch_images[idx]
        
        row = idx // n_cols
        col_base = (idx % n_cols) * 3
        
        # Convert to numpy
        original_np = original.cpu().permute(1, 2, 0).numpy()
        degraded_np = degraded.cpu().permute(1, 2, 0).numpy()
        corrected_np = corrected.cpu().permute(1, 2, 0).numpy()
        
        # Plot
        if n_rows == 1:
            axes[col_base].imshow(degraded_np.clip(0, 1))
            axes[col_base].set_title('Degraded', fontsize=10)
            axes[col_base].axis('off')
            
            axes[col_base + 1].imshow(corrected_np.clip(0, 1))
            axes[col_base + 1].set_title('Corrected', fontsize=10)
            axes[col_base + 1].axis('off')
            
            diff = np.abs(corrected_np - original_np) * 5
            axes[col_base + 2].imshow(diff.clip(0, 1))
            axes[col_base + 2].set_title('Error (5x)', fontsize=10)
            axes[col_base + 2].axis('off')
        else:
            axes[row, col_base].imshow(degraded_np.clip(0, 1))
            axes[row, col_base].set_title('Degraded', fontsize=10)
            axes[row, col_base].axis('off')
            
            axes[row, col_base + 1].imshow(corrected_np.clip(0, 1))
            axes[row, col_base + 1].set_title('Corrected', fontsize=10)
            axes[row, col_base + 1].axis('off')
            
            diff = np.abs(corrected_np - original_np) * 5
            axes[row, col_base + 2].imshow(diff.clip(0, 1))
            axes[row, col_base + 2].set_title('Error (5x)', fontsize=10)
            axes[row, col_base + 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved batch results to {save_path}")
    else:
        plt.show()
    
    plt.close()