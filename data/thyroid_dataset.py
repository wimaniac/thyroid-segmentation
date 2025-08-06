import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import os
from torch.utils.data import Dataset

class ThyroidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, rgb=False):
        self.image_paths = sorted([os.path.join(image_dir, fname) for fname in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)])
        self.transform = transform
        self.rgb = rgb

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.rgb:
            image = cv2.imread(self.image_paths[idx])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
            image = np.expand_dims(image, axis=-1)

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        mask = (mask > 0).astype(np.float32)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask'].unsqueeze(0)

        return image, mask, self.image_paths[idx]

def get_train_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ElasticTransform(alpha=90, sigma=5.0, p=0.3),
        A.RandomBrightnessContrast(p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(10, 30),
            hole_width_range=(10, 30),
            fill=0.0,
            p=0.3),
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ], additional_targets={'mask': 'mask'})

def get_val_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})

def get_test_transform():
    return A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=(0.5, 0.5, 0.5) , std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ], additional_targets={'mask': 'mask'})