import os
import glob
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

class SegmentationDataset(Dataset):
    """
    Generic segmentation dataset loader suitable for ACDC, Cityscapes, or dummy data.
    Assumes standard directory structure where images and masks can be matched
    by sorting or by replacing a substring in the filename.
    """
    def __init__(self, image_dir, mask_dir=None, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Glob all images (support nested directories)
        self.images = sorted(glob.glob(os.path.join(image_dir, '**/*.[pj][pn][g]'), recursive=True))
        if not self.images:
            # Fallback for flat dirs
            self.images = sorted(glob.glob(os.path.join(image_dir, '*.[pj][pn][g]')))
            
        self.masks = []
        if self.mask_dir is not None:
            # specifically search for ACDC/Cityscapes ground truth mask types to avoid colors/instance files
            self.masks = sorted(glob.glob(os.path.join(mask_dir, '**/*labelIds.png'), recursive=True))
            if not self.masks:
                self.masks = sorted(glob.glob(os.path.join(mask_dir, '**/*labelTrainIds.png'), recursive=True))
            if not self.masks:
                self.masks = sorted(glob.glob(os.path.join(mask_dir, '**/*.[pj][pn][g]'), recursive=True))
            if not self.masks:
                self.masks = sorted(glob.glob(os.path.join(mask_dir, '*.[pj][pn][g]')))
            
            # Simple assumption: masks and images are perfectly aligned by sorting
            assert len(self.images) == len(self.masks), f"Found {len(self.images)} images but {len(self.masks)} masks in {image_dir} and {mask_dir}!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask = None
        if self.mask_dir is not None:
            mask_path = self.masks[idx]
            # Masks are typically 1-channel images where pixel value = class index
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")

        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']

            if not torch.is_tensor(image):
                image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if mask is not None and not torch.is_tensor(mask):
                mask = torch.from_numpy(mask).long()

        else:
            # Fallback if no transform is provided: normalize to 0-1 and shape CHW
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            if mask is not None:
                mask = torch.from_numpy(mask).long()

        if mask is not None:
            return image, mask
        return image
