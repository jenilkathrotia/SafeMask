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
    parser.add_argument('--dummy',  action='store_true',
                        help='Generate and train on small dummy dataset.')
    return parser.parse_args()


def create_dummy_data(config):
    import cv2
    import numpy as np

    H, W        = config['dataset']['image_size']
    num_classes = config['dataset']['num_classes']

    for split in ['train', 'val']:
        img_dir  = config['dataset'][f'{split}_image_dir']
        mask_dir = config['dataset'][f'{split}_mask_dir']
        os.makedirs(img_dir,  exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)

        num_samples = 5 if split == 'train' else 2
        for i in range(num_samples):
            img  = np.zeros((H, W, 3), dtype=np.uint8)
            mask = np.zeros((H, W),    dtype=np.uint8)

            class_idx = (i % (num_classes - 1)) + 1
            start, end = 20 + i * 10, 100 + i * 10
            cv2.rectangle(img,  (start, start), (end, end), (255, 0, 0), -1)
            cv2.rectangle(mask, (start, start), (end, end), int(class_idx), -1)

            cv2.imwrite(os.path.join(img_dir,  f'img_{i}.png'), img)
            cv2.imwrite(os.path.join(mask_dir, f'img_{i}.png'), mask)


def build_transforms(img_size, augment=True):
    """
    Returns albumentations transform pipelines.

    Train: resize + realistic adverse-condition augmentations.
    Val  : resize only (no randomness).
    """
    H, W = img_size

    if augment:
        train_transform = A.Compose([
            A.Resize(height=H, width=W),
            # ---- Geometry ----
            A.HorizontalFlip(p=0.5),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1),
                     rotate=(-10, 10), p=0.4),
            # ---- Adverse-condition simulation ----
            A.RandomFog(fog_coef_range=(0.1, 0.4), p=0.3),
            A.RandomRain(blur_value=3, brightness_coefficient=0.9, p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.4),
            A.GaussNoise(p=0.3),
            A.MotionBlur(blur_limit=5, p=0.2),
        ])
    else:
        train_transform = A.Compose([A.Resize(height=H, width=W)])

    val_transform = A.Compose([A.Resize(height=H, width=W)])

    return train_transform, val_transform


def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.dummy:
        print("Generating dummy dataset for testing...")
        create_dummy_data(config)

    device = torch.device(
        'mps'  if torch.backends.mps.is_available()  else
        'cuda' if torch.cuda.is_available()           else
        'cpu'
    )
    print(f"Using device: {device}")

    img_size = config['dataset']['image_size']
    train_transform, val_transform = build_transforms(img_size, augment=not args.dummy)

    cs136_config = config.get('cs136_preprocessing', {})
    train_dataset = SegmentationDataset(
        config['dataset']['train_image_dir'],
        config['dataset']['train_mask_dir'],
        transform=train_transform,
        cs136_config=cs136_config,
    )
    val_dataset = SegmentationDataset(
        config['dataset']['val_image_dir'],
        config['dataset']['val_mask_dir'],
        transform=val_transform,
        cs136_config=cs136_config,
    )

    if len(train_dataset) == 0:
        print("No training images found! Please supply a dataset or run with --dummy.")
        return

    print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    num_workers = config['training'].get('num_workers', 0)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=num_workers
    )

    model     = create_model(config)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)   # ignore void label
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, device, config)
    trainer.train()


if __name__ == '__main__':
    main()