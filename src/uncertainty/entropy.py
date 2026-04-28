import torch
import torch.nn.functional as F
import numpy as np


def compute_entropy(logits):
    """
    Computes normalized pixel-wise Shannon entropy from model logits.

    Args:
        logits (torch.Tensor): Output logits from model of shape (B, C, H, W)

    Returns:
        normalized_entropy (torch.Tensor): Heatmap of shape (B, H, W), values 0-1.
    """
    probs = F.softmax(logits, dim=1)          # (B, C, H, W)
    eps = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)   # (B, H, W)
    num_classes = logits.shape[1]
    max_entropy = np.log(num_classes)
    return entropy / max_entropy


def compute_entropy_mc_dropout(model, image_tensor, n_passes=10):
    """
    MC Dropout uncertainty estimation.
    Runs inference N times with dropout ENABLED to get epistemic uncertainty.

    Args:
        model: PyTorch model (must have Dropout layers).
        image_tensor: (1, C, H, W) single image tensor on the correct device.
        n_passes (int): Number of stochastic forward passes.

    Returns:
        mean_entropy  (torch.Tensor): Shape (1, H, W) — averaged normalized entropy.
        mean_probs    (torch.Tensor): Shape (1, num_classes, H, W) — mean class probabilities.
        pred_variance (torch.Tensor): Shape (1, H, W) — pixel-wise variance across passes.
    """
    # ---- enable dropout at inference time ----
    model.train()          # activates Dropout layers
    with torch.no_grad():
        prob_list = []
        for _ in range(n_passes):
            logits = model(image_tensor)          # (1, C, H, W)
            probs  = F.softmax(logits, dim=1)     # (1, C, H, W)
            prob_list.append(probs.unsqueeze(0))  # (1, 1, C, H, W)

    model.eval()   # restore eval mode

    # stack → (n_passes, 1, C, H, W)
    stacked = torch.cat(prob_list, dim=0)          # (n_passes, 1, C, H, W)

    mean_probs = stacked.mean(dim=0)               # (1, C, H, W)
    # variance averaged over classes → scalar uncertainty per pixel
    pred_variance = stacked.var(dim=0).mean(dim=1) # (1, H, W)

    # entropy of the *mean* distribution
    eps = 1e-8
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1)
    num_classes = mean_probs.shape[1]
    mean_entropy = entropy / np.log(num_classes)   # (1, H, W)

    return mean_entropy, mean_probs, pred_variance


def get_warning_regions(normalized_entropy, threshold=0.5):
    """
    Returns a binary mask of high-uncertainty regions.

    Args:
        normalized_entropy (torch.Tensor): Shape (B, H, W)
        threshold (float): Pixels above this value are flagged.

    Returns:
        warning_mask (torch.Tensor): Binary mask (B, H, W)
    """
    return (normalized_entropy > threshold).float()