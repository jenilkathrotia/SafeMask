import torch
import torch.nn.functional as F
import numpy as np


def compute_entropy(logits):
    # Shannon entropy on the softmax, then divide by log(num_classes) to get 0..1.
    probs = F.softmax(logits, dim=1)
    eps = 1e-8
    entropy = -torch.sum(probs * torch.log(probs + eps), dim=1)
    num_classes = logits.shape[1]
    return entropy / np.log(num_classes)


def compute_entropy_mc_dropout(model, image_tensor, n_passes=10):
    # Run the model n_passes times with dropout on, average the probabilities,
    # then take entropy of the mean. Also returns per-pixel variance.
    model.train()  # enables dropout
    with torch.no_grad():
        prob_list = []
        for _ in range(n_passes):
            logits = model(image_tensor)
            probs = F.softmax(logits, dim=1)
            prob_list.append(probs.unsqueeze(0))
    model.eval()

    stacked = torch.cat(prob_list, dim=0)
    mean_probs = stacked.mean(dim=0)
    pred_variance = stacked.var(dim=0).mean(dim=1)

    eps = 1e-8
    entropy = -torch.sum(mean_probs * torch.log(mean_probs + eps), dim=1)
    num_classes = mean_probs.shape[1]
    mean_entropy = entropy / np.log(num_classes)

    return mean_entropy, mean_probs, pred_variance


def get_warning_regions(normalized_entropy, threshold=0.5):
    # Pixels above the threshold get marked as warning regions.
    return (normalized_entropy > threshold).float()