import os
import sys
import argparse
import yaml
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.segmentation_model import create_model
from src.uncertainty.entropy import compute_entropy, get_warning_regions
from src.visualization.visualize import plot_uncertainty_results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_weights', type=str, default='outputs/checkpoints/best_model.pth')
    parser.add_argument('--output', type=str, default='outputs/visualizations/result.png')
    return parser.parse_args()

def infer():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    
    model = create_model(config)
    if os.path.exists(args.model_weights):
        model.load_state_dict(torch.load(args.model_weights, map_location='cpu'))
        print(f"Loaded weights from {args.model_weights}")
    else:
        print(f"Warning: Weights not found at {args.model_weights}. Using randomly initialized model.")
    
    model.to(device)
    model.eval()
    
    # Load and preprocess image
    image = cv2.imread(args.image)
    if image is None:
        raise ValueError(f"Could not read image {args.image}")
    
    orig_h, orig_w = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img_size = config['dataset']['image_size']
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1])
    ])
    
    augmented = transform(image=image)
    img_tensor = augmented['image']
    img_tensor = torch.from_numpy(img_tensor.transpose(2, 0, 1)).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        
        # Simple argmax for baseline segmentation mask
        preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
        
        # Uncertainty
        entropy_tensor = compute_entropy(outputs)
        entropy = entropy_tensor.squeeze(0).cpu().numpy()
        
        threshold = config.get('uncertainty', {}).get('entropy_threshold', 0.5)
        warning_tensor = get_warning_regions(entropy_tensor, threshold)
        warning_mask = warning_tensor.squeeze(0).cpu().numpy()
        
    # Resize back to original
    preds_resized = cv2.resize(preds.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    entropy_resized = cv2.resize(entropy, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    warning_resized = cv2.resize(warning_mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    plot_uncertainty_results(image, preds_resized, entropy_resized, warning_resized, save_path=args.output)
    print(f"Saved prediction to {args.output}")

if __name__ == '__main__':
    infer()
