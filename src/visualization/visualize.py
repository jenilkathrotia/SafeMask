import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2

# -----------------------------------------------------------------------
# Official Cityscapes / ACDC 19-class color palette
# -----------------------------------------------------------------------
CITYSCAPES_COLORS = np.array([
    [128,  64, 128],   #  0 road
    [244,  35, 232],   #  1 sidewalk
    [ 70,  70,  70],   #  2 building
    [102, 102, 156],   #  3 wall
    [190, 153, 153],   #  4 fence
    [153, 153, 153],   #  5 pole
    [250, 170,  30],   #  6 traffic light
    [220, 220,   0],   #  7 traffic sign
    [107, 142,  35],   #  8 vegetation
    [152, 251, 152],   #  9 terrain
    [ 70, 130, 180],   # 10 sky
    [220,  20,  60],   # 11 person
    [255,   0,   0],   # 12 rider
    [  0,   0, 142],   # 13 car
    [  0,   0,  70],   # 14 truck
    [  0,  60, 100],   # 15 bus
    [  0,  80, 100],   # 16 train
    [  0,   0, 230],   # 17 motorcycle
    [119,  11,  32],   # 18 bicycle
], dtype=np.uint8)

CITYSCAPES_LABELS = [
    'road', 'sidewalk', 'building', 'wall', 'fence',
    'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain',
    'sky', 'person', 'rider', 'car', 'truck',
    'bus', 'train', 'motorcycle', 'bicycle',
]


def colorize_mask(mask, num_classes=19):
    """
    Converts a (H, W) integer class mask into an (H, W, 3) RGB image
    using the Cityscapes palette.
    """
    palette = CITYSCAPES_COLORS if num_classes == 19 else _random_palette(num_classes)
    mask    = np.clip(mask, 0, len(palette) - 1)
    return palette[mask]


def _random_palette(num_classes):
    rng = np.random.default_rng(42)
    return (rng.integers(0, 255, size=(num_classes, 3))).astype(np.uint8)


def _make_legend(present_classes, num_classes=19):
    """Returns matplotlib legend handles for classes visible in the mask."""
    if num_classes != 19:
        return []
    handles = []
    for cls in present_classes:
        if cls >= len(CITYSCAPES_LABELS):
            continue
        color  = CITYSCAPES_COLORS[cls] / 255.0
        handle = mpatches.Patch(color=color, label=CITYSCAPES_LABELS[cls])
        handles.append(handle)
    return handles


def plot_uncertainty_results(image, prediction, entropy, warning_mask,
                              save_path=None, num_classes=19):
    """
    Plots a 1×4 grid: original | segmentation mask | uncertainty heatmap | warning overlay.

    Args:
        image        (np.ndarray): RGB image (H, W, 3)
        prediction   (np.ndarray): Integer class mask (H, W)
        entropy      (np.ndarray): Normalized entropy heatmap (H, W), values 0–1
        warning_mask (np.ndarray): Binary warning mask (H, W)
        save_path    (str | None): Path to save figure; shows interactively if None
        num_classes  (int)       : Number of segmentation classes
    """
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    fig.patch.set_facecolor('#1e1e1e')
    for ax in axes:
        ax.set_facecolor('#1e1e1e')

    # ---- 1. Original image ----
    axes[0].imshow(image)
    axes[0].set_title("Original Image", color='white', fontsize=12, pad=8)
    axes[0].axis('off')

    # ---- 2. Segmentation mask with Cityscapes palette + legend ----
    colored_mask   = colorize_mask(prediction, num_classes)
    axes[1].imshow(colored_mask)
    axes[1].set_title("Segmentation Mask", color='white', fontsize=12, pad=8)
    axes[1].axis('off')

    present_classes = np.unique(prediction).tolist()
    legend_handles  = _make_legend(present_classes, num_classes)
    if legend_handles:
        axes[1].legend(
            handles=legend_handles,
            loc='lower left',
            fontsize=6,
            ncol=2,
            framealpha=0.6,
            facecolor='#1e1e1e',
            labelcolor='white'
        )

    # ---- 3. Uncertainty heatmap ----
    im3 = axes[2].imshow(entropy, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title("Uncertainty (Entropy)", color='white', fontsize=12, pad=8)
    axes[2].axis('off')
    cbar = fig.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    # ---- 4. Warning overlay ----
    overlay   = image.copy()
    red_mask  = np.zeros_like(overlay)
    red_mask[:, :, 0] = 255
    alpha = 0.5
    overlay[warning_mask == 1] = cv2.addWeighted(
        overlay[warning_mask == 1], 1 - alpha,
        red_mask[warning_mask == 1], alpha, 0
    )
    pct_uncertain = warning_mask.mean() * 100
    axes[3].imshow(overlay)
    axes[3].set_title(
        f"Warning Regions  ({pct_uncertain:.1f}% uncertain)",
        color='white', fontsize=12, pad=8
    )
    axes[3].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150,
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved visualization → {save_path}")
    else:
        plt.show()