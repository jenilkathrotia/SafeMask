import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as A

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.datasets.acdc_loader import SegmentationDataset
from src.models.segmentation_model import create_model
from src.training.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--dummy', action='store_true', help='Generate and train on small dummy dataset.')
    return parser.parse_args()

def create_dummy_data(config):
    import cv2
    import numpy as np
    
    H, W = config['dataset']['image_size']
    num_classes = config['dataset']['num_classes']
    
    for split in ['train', 'val']:
        img_dir = config['dataset'][f'{split}_image_dir']
        mask_dir = config['dataset'][f'{split}_mask_dir']
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        
        num_samples = 5 if split == 'train' else 2
        
        for i in range(num_samples):
            img = np.zeros((H, W, 3), dtype=np.uint8)
            mask = np.zeros((H, W), dtype=np.uint8)
            
            class_idx = (i % (num_classes - 1)) + 1
            start = 20 + i * 10
            end = 100 + i * 10
            cv2.rectangle(img, (start, start), (end, end), (255, 0, 0), -1)
            cv2.rectangle(mask, (start, start), (end, end), int(class_idx), -1)
            
            cv2.imwrite(os.path.join(img_dir, f'img_{i}.png'), img)
            cv2.imwrite(os.path.join(mask_dir, f'img_{i}.png'), mask)

def main():
    args = parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    if args.dummy:
        print("Generating dummy dataset for testing...")
        create_dummy_data(config)
        
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    img_size = config['dataset']['image_size']
    transform = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1])
    ])
    
    train_dataset = SegmentationDataset(config['dataset']['train_image_dir'], config['dataset']['train_mask_dir'], transform=transform)
    val_dataset = SegmentationDataset(config['dataset']['val_image_dir'], config['dataset']['val_mask_dir'], transform=transform)
    
    # Check if empty dataset, which could cause dataloader errors
    if len(train_dataset) == 0:
        print("No training images found! Please supply dataset or run with --dummy.")
        return

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['training'].get('num_workers', 0), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, num_workers=config['training'].get('num_workers', 0))
    
    model = create_model(config)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'], weight_decay=config['training']['weight_decay'])
    
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, config)
    trainer.train()

if __name__ == '__main__':
    main()
