import os
import cv2
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
import logging
import torch
from torch.distributions.beta import Beta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThyroidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, mask_suffix=".jpg", mixup_prob=0.1, alpha=0.1):
        """
        Dataset cho dữ liệu tuyến giáp.

        Args:
            image_dir (str): Đường dẫn đến thư mục chứa ảnh gốc.
            mask_dir (str): Đường dẫn đến thư mục chứa mask.
            transform: Transform áp dụng cho ảnh và mask.
            mixup_prob (float): Xác suất áp dụng MixUp (mặc định 0.1).
            alpha (float): Tham số phân phối Beta cho MixUp (mặc định 0.1).
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.mixup_prob = mixup_prob
        self.alpha = alpha
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg'))])
        logger.info(f"Found {len(self.images)} images in {image_dir}")
        if not self.images:
            logger.error(f"No images found in {image_dir}!")
            raise ValueError(f"No images found in {image_dir}")

        # Loại bỏ điều kiện ngưỡng, chỉ kiểm tra sự tồn tại của mask
        self.valid_pairs = []
        missing_masks = []
        intensities = []
        for img_name in self.images:
            base_name = os.path.splitext(img_name)[0]
            mask_name = img_name if self.mask_suffix == '.jpg' else base_name + self.mask_suffix
            mask_path = os.path.join(self.mask_dir, mask_name)
            if os.path.exists(mask_path):
                img_path = os.path.join(self.image_dir, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    image = image.astype(np.float32) / 255.0
                    intensities.append(image)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = (mask > 0).astype(np.float32)
                    self.valid_pairs.append((img_name, mask_name))
            else:
                missing_masks.append(img_name)
        if missing_masks:
            logger.warning(f"{len(missing_masks)} images missing masks: {missing_masks[:5]}")
            with open('missing_masks.txt', 'w') as f:
                f.write('\n'.join(missing_masks))
        if not self.valid_pairs:
            logger.error(f"No valid image-mask pairs found in {image_dir} and {mask_dir}!")
            raise ValueError(f"No valid image-mask pairs")
        logger.info(f"Found {len(self.valid_pairs)} valid image-mask pairs")

        # Tính mean và std từ intensities
        if intensities:
            intensities = np.concatenate([img.flatten() for img in intensities])
            self.mean = np.mean(intensities)
            self.std = np.std(intensities)
            logger.info(f"Calculated mean: {self.mean:.4f}, std: {self.std:.4f}")
        else:
            self.mean, self.std = 0.5, 0.5  # Giá trị mặc định nếu không có dữ liệu

    def __len__(self):
        return len(self.valid_pairs)

    def mixup_data(self, image1, mask1, image2, mask2, lambda_val):
        image_mixed = lambda_val * image1 + (1 - lambda_val) * image2
        mask_mixed = lambda_val * mask1 + (1 - lambda_val) * mask2
        return image_mixed, mask_mixed

    def __getitem__(self, idx):
        img_name, mask_name = self.valid_pairs[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            logger.error(f"Failed to load image or mask at {img_path} or {mask_path}")
            raise ValueError(f"Failed to load image or mask at {img_path} or {mask_path}")

        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)

        if np.random.random() < self.mixup_prob:
            idx2 = np.random.randint(0, len(self))
            img_name2, mask_name2 = self.valid_pairs[idx2]
            img_path2 = os.path.join(self.image_dir, img_name2)
            mask_path2 = os.path.join(self.mask_dir, mask_name2)

            image2 = cv2.imread(img_path2, cv2.IMREAD_GRAYSCALE)
            mask2 = cv2.imread(mask_path2, cv2.IMREAD_GRAYSCALE)

            if image2 is not None and mask2 is not None:
                image2 = image2.astype(np.float32) / 255.0
                mask2 = (mask2 > 0).astype(np.float32)

                if self.transform:
                    augmented2 = self.transform(image=image2, mask=mask2)
                    image2 = augmented2['image']
                    mask2 = augmented2['mask']
                    if len(mask2.shape) == 2:
                        mask2 = mask2.unsqueeze(0)

                beta_dist = Beta(self.alpha, self.alpha)
                lambda_val = beta_dist.sample().item()
                image, mask = self.mixup_data(image, mask, image2, mask2, lambda_val)

        return image, mask, img_name

def get_default_transform(train=True):
    """
    Trả về transform mặc định cho dataset.
    """
    if train:
        return A.Compose([
            A.Resize(256, 256),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
            A.RandomBrightnessContrast(brightness_limit=0.2, p=0.5),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(256, 256),
            A.RandomBrightnessContrast(p=0.1),
            ToTensorV2()
        ])

if __name__ == "__main__":
    train_image_dir = "data/processed/train/image"
    train_mask_dir = "data/processed/train/mask"
    val_image_dir = "data/processed/val/image"
    val_mask_dir = "data/processed/val/mask"

    train_transform = get_default_transform(train=True)
    val_transform = get_default_transform(train=False)

    train_dataset = ThyroidDataset(train_image_dir, train_mask_dir, transform=train_transform)
    val_dataset = ThyroidDataset(val_image_dir, val_mask_dir, transform=val_transform, mixup_prob=0.1, alpha=0.1)

    image, mask, img_name = val_dataset[0]
    logger.info(f"Image shape: {image.shape}, Mask shape: {mask.shape}, Image name: {img_name}")