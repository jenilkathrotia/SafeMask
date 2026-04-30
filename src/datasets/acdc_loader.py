import os
import glob
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

from src.preprocessing.cs136_preproc import (
    apply_cs136_preprocessing, detect_acdc_condition,
)


class SegmentationDataset(Dataset):
    """
    Generic segmentation dataset loader suitable for ACDC, Cityscapes, or dummy data.
    Assumes standard directory structure where images and masks can be matched
    by sorting or by replacing a substring in the filename.

    If ``cs136_config`` is provided (a dict from configs/config.yaml under
    ``cs136_preprocessing``), each image gets the CS 136 preprocessing
    applied right after loading and before albumentations: sigma=1 Gaussian,
    CLAHE on Lab-L for fog/night images, plus Canny edges as a 4th channel
    when ``canny_channel.enabled`` is true.
    """
    def __init__(self, image_dir, mask_dir=None, transform=None, cs136_config=None,
                 split=None):
        """``split`` (one of 'train', 'val', 'test') filters ACDC paths so
        a single image_dir at the rgb_anon root only returns frames from
        that split. Reference frames (*_rgb_ref_anon.png) are always skipped.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.cs136_config = cs136_config or {}
        self.split = split

        # Glob all images (support nested directories)
        self.images = sorted(glob.glob(os.path.join(image_dir, '**/*.[pj][pn][g]'), recursive=True))
        if not self.images:
            # Fallback for flat dirs
            self.images = sorted(glob.glob(os.path.join(image_dir, '*.[pj][pn][g]')))

        # ACDC: always drop clear-weather reference frames.
        self.images = [p for p in self.images if not p.endswith('_rgb_ref_anon.png')]
        # ACDC: optional split filter (path contains /train/ or /val/ or /test/).
        if self.split is not None:
            tag = f'/{self.split}/'
            self.images = [p for p in self.images if tag in p]

        self.masks = []
        if self.mask_dir is not None:
            # ACDC ground-truth: prefer labelTrainIds (already mapped to 0..18),
            # fall back to labelIds, then anything else.
            self.masks = sorted(glob.glob(os.path.join(mask_dir, '**/*labelTrainIds.png'), recursive=True))
            if not self.masks:
                self.masks = sorted(glob.glob(os.path.join(mask_dir, '**/*labelIds.png'), recursive=True))
            if not self.masks:
                self.masks = sorted(glob.glob(os.path.join(mask_dir, '**/*.[pj][pn][g]'), recursive=True))
            if not self.masks:
                self.masks = sorted(glob.glob(os.path.join(mask_dir, '*.[pj][pn][g]')))
            # Match the same split filter as images.
            if self.split is not None:
                tag = f'/{self.split}/'
                self.masks = [p for p in self.masks if tag in p]

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

        # Run albumentations FIRST (it expects 3-channel RGB; runs RandomFog,
        # MotionBlur, etc.). Then apply CS 136 preprocessing on the augmented
        # image so the Canny edge channel reflects what the model will see.
        if self.transform is not None:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented['image']
                mask = augmented['mask']
            else:
                augmented = self.transform(image=image)
                image = augmented['image']

        # CS 136 preprocessing: Gaussian + CLAHE + Canny channel + morph cleanup.
        # No-op when cs136_config is empty or disabled.
        if self.cs136_config.get("enabled", False):
            condition = detect_acdc_condition(img_path)
            if torch.is_tensor(image):
                image = (image.numpy().transpose(1, 2, 0) * 255.0).astype('uint8')
            image = apply_cs136_preprocessing(image, condition, self.cs136_config)

        # Convert to CHW tensor (handles 3 or 4 channels).
        if not torch.is_tensor(image):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        if mask is not None and not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()

        if mask is not None:
            return image, mask
        return image
