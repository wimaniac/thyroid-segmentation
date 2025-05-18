import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
import albumentations as A
from albumentations.core.composition import Compose

# Định nghĩa đường dẫn
image_dir = r"../raw/Thyroid Dataset/tg3k/thyroid-image"
mask_dir = r"../raw/Thyroid Dataset/tg3k/thyroid-mask"
output_image_dir = r"../processed/image"
output_mask_dir = r"../processed/mask"

# Tạo thư mục đầu ra
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_image_dir, split), exist_ok=False)
    os.makedirs(os.path.join(output_mask_dir, split), exist_ok=False)

# Kích thước chuẩn hóa
TARGET_SIZE = (256, 256)

# Định nghĩa pipeline tăng cường dữ liệu (chỉ áp dụng cho tập train)
train_augmentation = Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=20, p=0.3),  # Giảm góc xoay để tránh làm mất ngữ cảnh tuyến tụy
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
    A.GaussNoise(p=0.2),  # Giảm nhiễu để giữ chi tiết
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
], additional_targets={'mask': 'mask'})


# Hàm padding để giữ tỷ lệ
def pad_image(image, target_size):
    h, w = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_h / h, target_w / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h),
                         interpolation=cv2.INTER_LINEAR if len(image.shape) == 3 else cv2.INTER_NEAREST)

    # Tạo nền đen
    padded = np.zeros((target_h, target_w, 3) if len(image.shape) == 3 else (target_h, target_w), dtype=image.dtype)
    top = (target_h - new_h) // 2
    left = (target_w - new_w) // 2
    if len(image.shape) == 3:
        padded[top:top + new_h, left:left + new_w, :] = resized
    else:
        padded[top:top + new_h, left:left + new_w] = resized
    return padded


# Hàm chuẩn hóa giá trị pixel
def normalize_image(image):
    return image.astype(np.float32) / 255.0


def binarize_mask(mask):
    _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


# Hàm xử lý và lưu dữ liệu
def process_and_save(files, split, image_input_dir, mask_input_dir, image_output_dir, mask_output_dir):
    for file in tqdm(files, desc=f"Processing {split}"):
        # Đọc ảnh và mask (cùng tên file)
        image_path = os.path.join(image_input_dir, file)
        mask_path = os.path.join(mask_input_dir, file)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Kiểm tra dữ liệu hợp lệ
        if image is None or mask is None:
            print(f"Warning: Skipping invalid file {file}")
            continue

        # Padding để giữ tỷ lệ
        image = pad_image(image, TARGET_SIZE)
        mask = pad_image(mask, TARGET_SIZE)

        # Chuẩn hóa
        image = normalize_image(image)
        mask = binarize_mask(mask)

        # Tăng cường dữ liệu (chỉ cho tập train)
        if split == 'train':
            augmented = train_augmentation(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        # Chuyển đổi lại sang định dạng uint8 để lưu
        image = (image * 255).astype(np.uint8)
        mask = mask * 255  # Lưu mask với giá trị 0 và 255

        # Lưu ảnh và mask
        cv2.imwrite(os.path.join(image_output_dir, split, file), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(mask_output_dir, split, file), mask)


# Lấy danh sách file ảnh
image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

# Kiểm tra số lượng
if len(image_files) != 3585:
    print(f"Warning: Found {len(image_files)} images, expected 3585.")

# Kiểm tra sự tồn tại của mask tương ứng
for file in image_files:
    if not os.path.exists(os.path.join(mask_dir, file)):
        print(f"Warning: Mask for {file} not found.")

# Chia tập dữ liệu (80:10:10)
train_files, temp_files = train_test_split(image_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

# Xử lý và lưu dữ liệu
process_and_save(train_files, 'train', image_dir, mask_dir, output_image_dir, output_mask_dir)
process_and_save(val_files, 'val', image_dir, mask_dir, output_image_dir, output_mask_dir)
process_and_save(test_files, 'test', image_dir, mask_dir, output_image_dir, output_mask_dir)

# Thống kê số lượng file
print("Tiền xử lý hoàn tất! Số lượng file trong các tập:")
print(f"- Train: {len(train_files)} ảnh/mask")
print(f"- Val: {len(val_files)} ảnh/mask")
print(f"- Test: {len(test_files)} ảnh/mask")
print("Dữ liệu đã được lưu vào:")
print(f"- data/processed/image/train, val, test")
print(f"- data/processed/mask/train, val, test")