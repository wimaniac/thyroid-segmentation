import os
import shutil
from sklearn.model_selection import train_test_split
import argparse


def create_directories(base_dir):
    """Tạo các thư mục train, val và các thư mục con image, mask."""
    dirs = [
        os.path.join(base_dir, "train", "image"),
        os.path.join(base_dir, "train", "mask"),
        os.path.join(base_dir, "val", "image"),
        os.path.join(base_dir, "val", "mask"),
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print(f"Created directories: {dirs}")


def get_paired_files(image_dir, mask_dir, mask_suffix="_mask.png"):
    """Lấy danh sách các cặp ảnh và mask, kiểm tra tính hợp lệ."""
    # Kiểm tra thư mục tồn tại
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory {image_dir} does not exist!")
    if not os.path.exists(mask_dir):
        raise ValueError(f"Mask directory {mask_dir} does not exist!")

    # Lấy danh sách file ảnh
    supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(supported_extensions)])

    if not image_files:
        raise ValueError(f"No image files found in {image_dir}!")

    paired_files = []
    missing_masks = []
    # Thử nhiều hậu tố mask
    mask_suffixes = [mask_suffix] if mask_suffix else ['_mask.png', '_mask.jpg', '.png', '.jpg', '_mask.tif']

    for img in image_files:
        found = False
        base_name = os.path.splitext(img)[0]
        for suffix in mask_suffixes:
            if suffix.startswith('_mask'):
                mask = base_name + suffix
            else:
                mask = base_name + suffix if suffix.startswith('.') else img
            mask_path = os.path.join(mask_dir, mask)
            if os.path.exists(mask_path):
                paired_files.append((img, mask))
                found = True
                break
        if not found:
            missing_masks.append(img)

    if missing_masks:
        print(f"Warning: {len(missing_masks)} images missing masks:")
        print(f"Sample missing: {missing_masks[:5]}")
        with open('missing_masks.txt', 'w') as f:
            f.write('\n'.join(missing_masks))
        print(f"Full list saved to missing_masks.txt")

    if not paired_files:
        print(f"Image files found: {len(image_files)}")
        print(f"Sample images: {image_files[:5]}")
        raise ValueError(f"No paired image-mask files found! Tried suffixes: {mask_suffixes}")

    print(f"Found {len(paired_files)} paired image-mask files")
    return paired_files


def split_and_copy_files(paired_files, train_dir, val_dir, train_ratio=0.8, random_seed=42):
    """Chia và sao chép file vào thư mục train, val."""
    images, masks = zip(*paired_files)
    train_images, val_images, train_masks, val_masks = train_test_split(
        images, masks, train_size=train_ratio, random_state=random_seed, shuffle=True
    )

    # Sao chép file vào thư mục train
    for img, mask in zip(train_images, train_masks):
        src_img = os.path.join(args.trainval_image_dir, img)
        src_mask = os.path.join(args.trainval_mask_dir, mask)
        dst_img = os.path.join(train_dir, "image", img)
        dst_mask = os.path.join(train_dir, "mask", mask)
        shutil.copy(src_img, dst_img)
        shutil.copy(src_mask, dst_mask)

    # Sao chép file vào thư mục val
    for img, mask in zip(val_images, val_masks):
        src_img = os.path.join(args.trainval_image_dir, img)
        src_mask = os.path.join(args.trainval_mask_dir, mask)
        dst_img = os.path.join(val_dir, "image", img)
        dst_mask = os.path.join(val_dir, "mask", mask)
        shutil.copy(src_img, dst_img)
        shutil.copy(src_mask, dst_mask)

    print(f"Train: {len(train_images)} images, {len(train_masks)} masks")
    print(f"Validation: {len(val_images)} images, {len(val_masks)} masks")


def main(args):
    # Kiểm tra thư mục đầu vào
    print(f"Checking input directories:")
    print(f"Trainval image dir: {args.trainval_image_dir}")
    print(f"Trainval mask dir: {args.trainval_mask_dir}")
    print(f"Using mask suffix: {args.mask_suffix if args.mask_suffix else 'auto-detect'}")

    # Tạo thư mục
    create_directories(args.data_dir)

    # Lấy danh sách cặp ảnh-mask
    paired_files = get_paired_files(args.trainval_image_dir, args.trainval_mask_dir, args.mask_suffix)

    # Chia và sao chép file
    split_and_copy_files(
        paired_files,
        train_dir=os.path.join(args.data_dir, "train"),
        val_dir=os.path.join(args.data_dir, "val"),
        train_ratio=args.train_ratio,
        random_seed=args.random_seed
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split trainval data into train and validation sets")
    parser.add_argument('--data_dir', type=str, default='data/processed', help='Base directory for processed data')
    parser.add_argument('--trainval_image_dir', type=str, default='data/tn3k/trainval-image',
                        help='Directory of trainval images')
    parser.add_argument('--trainval_mask_dir', type=str, default='data/tn3k/trainval-mask',
                        help='Directory of trainval masks')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='Ratio of data for training (0-1)')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--mask_suffix', type=str, default=None,
                        help='Suffix for mask files (e.g., _mask.png, .jpg). If None, auto-detect.')
    args = parser.parse_args()

    main(args)