import torch
import torch.nn.functional as F
import numpy as np

def compute_entropy(logits):
    """
    Computes normalized pixel-wise Shannon entropy from model logits.
    
    Args:
        logits (torch.Tensor): Output logits from model of shape (B, C, H, W)
        
    Returns:
        normalized_entropy (torch.Tensor): Heatmap of shape (B, H, W)
                                           Values are between 0 and 1.
    """
    # 1. Convert logits to probabilities
    probs = F.softmax(logits, dim=1) # (B, C, H, W)
    
    # 2. Compute Shannon entropy: H = -sum(p * log(p))
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1) # (B, H, W)
    
    # 3. Normalize by maximum possible entropy (log(C))
    num_classes = logits.shape[1]
    max_entropy = np.log(num_classes)
    
    normalized_entropy = entropy / max_entropy
    
    return normalized_entropy

def get_warning_regions(normalized_entropy, threshold=0.5):
    """
    Returns a binary mask of regions where uncertainty is high.
    
    Args:
        normalized_entropy (torch.Tensor): Normalized entropy of shape (B, H, W)
        threshold (float): Threshold above which a pixel is considered uncertain.
        
    Returns:
        warning_mask (torch.Tensor): Binary mask (B, H, W)
    """
    warning_mask = (normalized_entropy > threshold).float()
    return warning_mask
