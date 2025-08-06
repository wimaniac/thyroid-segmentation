import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def compute_fg_ratio(image_dir, mask_dir, mask_suffix=".jpg"):
    fg_ratios = []
    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.jpg')])
    for img_name in image_files:
        mask_path = os.path.join(mask_dir, img_name.replace(".jpg", mask_suffix))
        if not os.path.exists(mask_path):
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        mask_bin = (mask > 127).astype(np.uint8)
        fg_ratio = mask_bin.mean()
        fg_ratios.append(fg_ratio)
    return np.array(fg_ratios)

# Đường dẫn tương ứng với config của bạn
train_image_dir = "data/processed/train/image"
train_mask_dir  = "data/processed/train/mask"
val_image_dir   = "data/processed/val/image"
val_mask_dir    = "data/processed/val/mask"
mask_suffix     = ".jpg"

train_fg_ratios = compute_fg_ratio(train_image_dir, train_mask_dir, mask_suffix)
val_fg_ratios = compute_fg_ratio(val_image_dir, val_mask_dir, mask_suffix)

# Plot histogram
plt.figure(figsize=(12, 5))
plt.hist(train_fg_ratios, bins=40, alpha=0.7, label='Train', color='blue')
plt.hist(val_fg_ratios, bins=40, alpha=0.7, label='Validation', color='orange')
plt.xlabel('Foreground Ratio')
plt.ylabel('Number of Images')
plt.title('Distribution of Foreground Ratio (After Filtering)')
plt.legend()
plt.grid(True)
plt.show()

# In thông tin thống kê cơ bản
print(f"Train: min={train_fg_ratios.min():.4f}, max={train_fg_ratios.max():.4f}, mean={train_fg_ratios.mean():.4f}")
print(f"Val:   min={val_fg_ratios.min():.4f}, max={val_fg_ratios.max():.4f}, mean={val_fg_ratios.mean():.4f}")
