import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
import albumentations as A

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.segmentation_model import create_model
from src.preprocessing.cs136_preproc import (
    apply_cs136_preprocessing,
    detect_acdc_condition,
)
from src.uncertainty.entropy import (
    compute_entropy,
    compute_entropy_mc_dropout,
    get_warning_regions,
)
from src.visualization.visualize import plot_uncertainty_results


def parse_args():
    parser = argparse.ArgumentParser(description="SafeMask inference script")
    parser.add_argument('--config',        type=str,  default='configs/config.yaml')
    parser.add_argument('--image',         type=str,  required=True,
                        help='Path to input image')
    parser.add_argument('--model_weights', type=str,  default='outputs/checkpoints/best_model.pth')
    parser.add_argument('--output',        type=str,  default='outputs/visualizations/result.png')
    parser.add_argument('--mc_dropout',    action='store_true',
                        help='Use MC Dropout for uncertainty instead of single-pass entropy.')
    parser.add_argument('--mc_passes',     type=int,  default=None,
                        help='Number of MC Dropout passes (overrides config).')
    return parser.parse_args()


def infer():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device(
        'mps'  if torch.backends.mps.is_available()  else
        'cuda' if torch.cuda.is_available()           else
        'cpu'
    )
    print(f"Device: {device}")

    model = create_model(config)
    if os.path.exists(args.model_weights):
        model.load_state_dict(torch.load(args.model_weights, map_location='cpu'))
        print(f"Loaded weights from {args.model_weights}")
    else:
        print(f"⚠️  Weights not found at {args.model_weights}. Using random init.")

    model.to(device)
    model.eval()

    # ---- Load & preprocess ----
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    img_size  = config['dataset']['image_size']
    transform = A.Compose([A.Resize(height=img_size[0], width=img_size[1])])
    augmented = transform(image=image)
    proc_image = augmented['image']

    # Apply the same CS 136 preprocessing the model trained with so the input
    # has the right number of channels (3 or 4).
    cs136_cfg = config.get('cs136_preprocessing', {})
    if cs136_cfg.get('enabled', False):
        condition = detect_acdc_condition(args.image)
        proc_image = apply_cs136_preprocessing(proc_image, condition, cs136_cfg)

    img_tensor = (
        torch.from_numpy(proc_image.transpose(2, 0, 1))
        .float() / 255.0
    ).unsqueeze(0).to(device)

    threshold = config.get('uncertainty', {}).get('entropy_threshold', 0.5)

    # ---- Inference ----
    with torch.no_grad():
        if args.mc_dropout:
            n_passes = args.mc_passes or config.get('uncertainty', {}).get('mc_dropout_passes', 10)
            print(f"Running MC Dropout ({n_passes} passes)…")
            entropy_tensor, mean_probs, variance = compute_entropy_mc_dropout(
                model, img_tensor, n_passes=n_passes
            )
            preds = torch.argmax(mean_probs, dim=1).squeeze(0).cpu().numpy()
            print(f"  Mean pixel variance: {variance.mean().item():.5f}")
        else:
            outputs        = model(img_tensor)
            preds          = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
            entropy_tensor = compute_entropy(outputs)

    entropy      = entropy_tensor.squeeze(0).cpu().numpy()
    warning_mask = get_warning_regions(entropy_tensor, threshold).squeeze(0).cpu().numpy()

    # ---- Resize back to original resolution ----
    preds_r   = cv2.resize(preds.astype(np.uint8),   (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    entropy_r = cv2.resize(entropy,                   (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    warning_r = cv2.resize(warning_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    uncertain_pct = warning_r.mean() * 100
    method = "MC Dropout" if args.mc_dropout else "Softmax Entropy"
    print(f"Uncertainty method : {method}")
    print(f"Uncertain pixels   : {uncertain_pct:.1f}%")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plot_uncertainty_results(
        image, preds_r, entropy_r, warning_r,
        save_path=args.output,
        num_classes=config['dataset']['num_classes']
    )


if __name__ == '__main__':
    infer()