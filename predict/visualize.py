import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv2

import numpy as np
import matplotlib.pyplot as plt
from myconfig1 import get_config

def visualize_inference_results(test_image_dir, test_mask_dir, pred_dir, save_dir):
    # Lấy danh sách file từ các thư mục
    image_files = sorted([f for f in os.listdir(test_image_dir) if f.endswith('.jpg') or f.endswith('.png')])
    mask_files = sorted([f for f in os.listdir(test_mask_dir) if f.endswith('.jpg') or f.endswith('.png')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.jpg') or f.endswith('.png')])

    # Đảm bảo số lượng file khớp
    if len(image_files) != len(mask_files) or len(image_files) != len(pred_files):
        print(f"Warning: Number of images ({len(image_files)}), ground truth masks ({len(mask_files)}), and predicted masks ({len(pred_files)}) do not match.")
        min_len = min(len(image_files), len(mask_files), len(pred_files))
        image_files = image_files[:min_len]
        mask_files = mask_files[:min_len]
        pred_files = pred_files[:min_len]

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)

    # Trực quan hóa từng bộ ba
    for idx in range(len(image_files)):
        # Đường dẫn file
        image_path = os.path.join(test_image_dir, image_files[idx])
        mask_path = os.path.join(test_mask_dir, mask_files[idx])
        pred_path = os.path.join(pred_dir, pred_files[idx])

        # Đọc ảnh
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None or pred_mask is None:
            print(f"Error loading image/mask at index {idx}: {image_path}, {mask_path}, or {pred_path}")
            continue

        # Chuẩn hóa kích thước nếu cần
        if image.shape[:2] != mask.shape or image.shape[:2] != pred_mask.shape:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            pred_mask = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))

        # Tạo hình ảnh so sánh
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Ground Truth Mask')
        axes[1].axis('off')
        axes[2].imshow(pred_mask, cmap='gray')
        axes[2].set_title('Predicted Mask')
        axes[2].axis('off')
        plt.suptitle(f"Comparison - {image_files[idx]}")

        # Lưu hình ảnh
        plt.savefig(os.path.join(save_dir, f"comparison_{image_files[idx].split('.')[0]}.png"))
        plt.close()

if __name__ == "__main__":
    args = get_config()

    visualize_inference_results(
        test_image_dir=args.test_image_dir,
        test_mask_dir=args.test_mask_dir,
        pred_dir=args.pred_dir,
        save_dir=args.save_dir
    )
    print(f"Inference visualization completed. Images saved to {args.save_dir}.")