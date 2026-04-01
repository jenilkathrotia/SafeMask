import os
import sys
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.acdc_loader import SegmentationDataset
from src.models.segmentation_model import create_model
from src.uncertainty.entropy import compute_entropy, get_warning_regions
from src.evaluation.metrics import compute_iou, compute_pixel_accuracy, compute_uncertainty_metrics

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--condition', type=str, default='val', help='Dataset split/condition to evaluate (e.g., clear, fog, rain)')
    parser.add_argument('--model_weights', type=str, default='outputs/checkpoints/best_model.pth')
    return parser.parse_args()

def evaluate():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating on {device}")
    
    # Simple logic to point to different test sets based on condition
    # Fallback to val if specific condition dir doesn't exist in config
    img_dir_key = f'{args.condition}_image_dir'
    mask_dir_key = f'{args.condition}_mask_dir'
    
    img_dir = config['dataset'].get(img_dir_key, config['dataset']['val_image_dir'])
    mask_dir = config['dataset'].get(mask_dir_key, config['dataset']['val_mask_dir'])
    
    img_size = config['dataset']['image_size']
    transform = A.Compose([A.Resize(height=img_size[0], width=img_size[1])])
    
    dataset = SegmentationDataset(img_dir, mask_dir, transform=transform)
    if len(dataset) == 0:
        print(f"No datset found at {img_dir}. Exiting.")
        return
        
    loader = DataLoader(dataset, batch_size=config['training'].get('batch_size', 4), shuffle=False)
    
    model = create_model(config)
    if os.path.exists(args.model_weights):
        model.load_state_dict(torch.load(args.model_weights, map_location='cpu'))
    
    model.to(device)
    model.eval()
    
    num_classes = config['dataset']['num_classes']
    threshold = config.get('uncertainty', {}).get('entropy_threshold', 0.5)
    
    total_iou = np.zeros(num_classes)
    valid_classes = np.zeros(num_classes)
    total_acc = 0.0
    
    total_uncert_correct = 0.0
    total_uncert_incorrect = 0.0
    total_warn_prec = 0.0
    total_warn_rec = 0.0
    
    batches = len(loader)
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Evaluating ({args.condition})"):
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Semantic segmentations metrics
            ious = compute_iou(preds, masks, num_classes)
            for i, iou in enumerate(ious):
                if not np.isnan(iou):
                    total_iou[i] += iou
                    valid_classes[i] += 1
            
            acc = compute_pixel_accuracy(preds, masks)
            total_acc += acc
            
            # Uncertainty
            entropy = compute_entropy(outputs)
            warning_mask = get_warning_regions(entropy, threshold)
            
            uncert_metrics = compute_uncertainty_metrics(preds, masks, entropy, warning_mask)
            total_uncert_correct += uncert_metrics['avg_entropy_correct']
            total_uncert_incorrect += uncert_metrics['avg_entropy_incorrect']
            total_warn_prec += uncert_metrics['warning_precision']
            total_warn_rec += uncert_metrics['warning_recall']
            
    # Aggregate
    valid_classes[valid_classes == 0] = 1 # avoid div by zero
    class_ious = total_iou / valid_classes
    mIoU = np.nanmean(class_ious)
    
    print("\n--- Evaluation Results ---")
    print(f"Condition: {args.condition}")
    print(f"mIoU: {mIoU:.4f}")
    print(f"Pixel Accuracy: {total_acc / batches:.4f}")
    print(f"Avg Entropy (Correct Pixels): {total_uncert_correct / batches:.4f}")
    print(f"Avg Entropy (Incorrect Pixels): {total_uncert_incorrect / batches:.4f}")
    print(f"Warning Region Precision: {total_warn_prec / batches:.4f}")
    print(f"Warning Region Recall: {total_warn_rec / batches:.4f}")

if __name__ == '__main__':
    evaluate()
