import os
import shutil

# Danh sách file ảnh cần chuyển lại
img_list = [
    "0450.jpg", "0535.jpg", "1099.jpg",
    "1317.jpg", "1725.jpg", "2319.jpg", "2487.jpg"
]

# Thư mục
val_img_dir = "data/processed/val/image"
val_mask_dir = "data/processed/val/mask"

train_img_dir = "data/processed/train/image"
train_mask_dir = "data/processed/train/mask"

# Đảm bảo thư mục val tồn tại
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(val_mask_dir, exist_ok=True)

for img_name in img_list:
    # Đường dẫn ngược lại
    src_img = os.path.join(train_img_dir, img_name)
    src_mask = os.path.join(train_mask_dir, img_name)
    dst_img = os.path.join(val_img_dir, img_name)
    dst_mask = os.path.join(val_mask_dir, img_name)

    # Chuyển ảnh
    if os.path.exists(src_img):
        shutil.move(src_img, dst_img)
        print(f"Đã chuyển {src_img} -> {dst_img}")
    else:
        print(f"Không tìm thấy ảnh {src_img}")

    # Chuyển mask
    if os.path.exists(src_mask):
        shutil.move(src_mask, dst_mask)
        print(f"Đã chuyển {src_mask} -> {dst_mask}")
    else:
        print(f"Không tìm thấy mask {src_mask}")
