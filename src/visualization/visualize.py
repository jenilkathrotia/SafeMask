import os
import matplotlib.pyplot as plt
import numpy as np
import cv2

def plot_uncertainty_results(image, prediction, entropy, warning_mask, save_path=None):
    """
    Plots a 1x4 grid comparing the original image, segmentation mask, uncertainty heatmap,
    and warning overlay.
    
    Args:
        image (np.ndarray): Original image (H, W, 3) in RGB
        prediction (np.ndarray): Predicted segmentation mask (H, W)
        entropy (np.ndarray): Normalized entropy heatmap (H, W)
        warning_mask (np.ndarray): Binary warning mask (H, W)
        save_path (str): Path to save the figure
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # 2. Segmentation Mask
    axes[1].imshow(prediction, cmap='tab20')
    axes[1].set_title("Segmentation Mask")
    axes[1].axis('off')
    
    # 3. Uncertainty Heatmap
    im3 = axes[2].imshow(entropy, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Uncertainty (Entropy)")
    axes[2].axis('off')
    fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    # 4. Warning Overlay (Red for uncertain regions)
    overlay = image.copy()
    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255  # Red channel
    
    # Blend red where warning_mask is 1
    alpha = 0.5
    overlay[warning_mask == 1] = cv2.addWeighted(
        overlay[warning_mask == 1], 1 - alpha,
        red_mask[warning_mask == 1], alpha, 0
    )
    
    axes[3].imshow(overlay)
    axes[3].set_title("Warning Regions (Thresholded)")
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()
