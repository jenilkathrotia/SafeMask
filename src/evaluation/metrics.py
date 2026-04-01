import numpy as np

def compute_iou(pred, target, num_classes):
    """
    Computes Intersection over Union (IoU) per class and mIoU.
    """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        
        if union == 0:
            ious.append(float('nan'))  # Class not present in this sample
        else:
            ious.append(float(intersection) / float(max(union, 1)))
            
    return np.array(ious)

def compute_pixel_accuracy(pred, target):
    """
    Computes global pixel accuracy.
    """
    correct = (pred == target).sum().item()
    total = target.numel()
    return correct / total

def compute_uncertainty_metrics(pred, target, entropy, warning_mask):
    """
    Evaluates how well the uncertainty correlates with errors.
    
    Args:
        pred, target: (B, H, W)
        entropy: (B, H, W)
        warning_mask: (B, H, W) binary
        
    Returns:
        dict with metrics
    """
    correct_mask = (pred == target)
    incorrect_mask = ~correct_mask
    
    avg_entropy_correct = entropy[correct_mask].mean().item() if correct_mask.any() else 0.0
    avg_entropy_incorrect = entropy[incorrect_mask].mean().item() if incorrect_mask.any() else 0.0
    
    # Overlap between warning regions and misclassified pixels
    # Precision of warning regions: P(incorrect | warning)
    warning_pixels = warning_mask.sum().item()
    correctly_warned = (warning_mask.bool() & incorrect_mask).sum().item()
    
    warning_precision = correctly_warned / warning_pixels if warning_pixels > 0 else 0.0
    
    # Recall of warning regions: P(warning | incorrect)
    total_incorrect = incorrect_mask.sum().item()
    warning_recall = correctly_warned / total_incorrect if total_incorrect > 0 else 0.0
    
    return {
        'avg_entropy_correct': avg_entropy_correct,
        'avg_entropy_incorrect': avg_entropy_incorrect,
        'warning_precision': warning_precision,
        'warning_recall': warning_recall
    }
