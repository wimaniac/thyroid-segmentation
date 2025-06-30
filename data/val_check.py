import numpy as np
from thyroid_dataset import ThyroidDataset, get_default_transform
import os
import logging
import torch
# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Định nghĩa đường dẫn
val_image_dir = "data/processed/val/image"
val_mask_dir = "data/processed/val/mask"
train_image_dir = "data/processed/train/image"
train_mask_dir = "data/processed/train/mask"

# Lấy transform mặc định
val_transform = get_default_transform(train=False)

# Khởi tạo dataset cho validation
val_dataset = ThyroidDataset(image_dir=val_image_dir, mask_dir=val_mask_dir, transform=val_transform)
logger.info(f"Tổng số ảnh trong tập validation: {len(val_dataset.images)}")
logger.info(f"Số cặp image-mask hợp lệ trong tập validation: {len(val_dataset.valid_pairs)}")

# Tìm ảnh thiếu mask trong tập validation
missing_val_masks = [img_name for img_name in val_dataset.images if img_name not in [pair[0] for pair in val_dataset.valid_pairs]]
if missing_val_masks:
    logger.info(f"Số ảnh thiếu mask trong tập validation: {len(missing_val_masks)}")
    logger.info(f"Danh sách các ảnh thiếu mask trong tập validation: {missing_val_masks}")
else:
    logger.info("Không có ảnh nào thiếu mask trong tập validation.")

# Khởi tạo dataset cho train
train_dataset = ThyroidDataset(image_dir=train_image_dir, mask_dir=train_mask_dir, transform=val_transform)
logger.info(f"Tổng số ảnh trong tập train: {len(train_dataset.images)}")
logger.info(f"Số cặp image-mask hợp lệ trong tập train: {len(train_dataset.valid_pairs)}")

# Tìm ảnh thiếu mask trong tập train
missing_train_masks = [img_name for img_name in train_dataset.images if img_name not in [pair[0] for pair in train_dataset.valid_pairs]]
if missing_train_masks:
    logger.info(f"Số ảnh thiếu mask trong tập train: {len(missing_train_masks)}")
    logger.info(f"Danh sách các ảnh thiếu mask trong tập train: {missing_train_masks}")
else:
    logger.info("Không có ảnh nào thiếu mask trong tập train.")

# Tính thống kê intensity và mask ratio
intensities = []
mask_ratios = []
for img, mask, _ in val_dataset:
    img = img.numpy().flatten() if isinstance(img, torch.Tensor) else img.flatten()
    mask = mask.numpy().flatten() if isinstance(mask, torch.Tensor) else mask.flatten()
    intensities.append(np.mean(img))
    mask_ratios.append(np.mean(mask))

logger.info(f"Mean intensity: {np.mean(intensities):.4f}, Std intensity: {np.std(intensities):.4f}")
logger.info(f"Mean mask ratio: {np.mean(mask_ratios):.4f}, Std mask ratio: {np.std(mask_ratios):.4f}")