import os
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import glob

def get_unique_filename(base_path):
    """Tạo tên file duy nhất bằng cách thêm số thứ tự nếu file đã tồn tại."""
    base, ext = os.path.splitext(base_path)
    counter = 1
    new_path = base_path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path

def get_unique_dir(base_dir):
    """Tạo tên thư mục duy nhất bằng cách thêm số thứ tự nếu thư mục đã tồn tại."""
    counter = 1
    new_dir = base_dir
    while os.path.exists(new_dir):
        new_dir = f"{base_dir}_{counter}"
        counter += 1
    return new_dir

def find_latest_pred_dir(base_pred_dir):
    """Tìm thư mục dự đoán mới nhất (hỗ trợ hậu tố như pretrained_unet_1)."""
    pattern = os.path.join("results", "predictions", f"{base_pred_dir}*")
    pred_dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return pred_dirs[0] if pred_dirs else os.path.join("results", "predictions", base_pred_dir)

def visualize_prediction(image_path, mask_path=None, pred_path=None, save_path=None):
    """Hiển thị hoặc lưu ảnh gốc, mask thật, và mask dự đoán."""
    image = np.array(Image.open(image_path).convert("L"))
    mask = np.array(Image.open(mask_path).convert("L")) if mask_path and os.path.exists(mask_path) else None
    pred = np.array(Image.open(pred_path).convert("L")) if pred_path and os.path.exists(pred_path) else None

    num_plots = 3 if mask is not None and pred is not None else 2 if pred is not None or mask is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(4 * num_plots, 5))

    if num_plots == 1:
        axes = [axes]

    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    if mask is not None:
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Ground Truth Mask")
        axes[1].axis('off')

    if pred is not None:
        idx = 2 if mask is not None else 1
        axes[idx].imshow(pred, cmap='gray')
        axes[idx].set_title("Predicted Mask")
        axes[idx].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_path = get_unique_filename(save_path)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Visualize predictions for different models")
    parser.add_argument('--model', type=str, default=None, help='Model name (unet, pretrained_unet, torchvision_unet)')
    parser.add_argument('--pred_dir', type=str, default=None, help='Path to directory containing predicted masks')
    parser.add_argument('--num_images', type=int, default=5, help='Number of images to visualize')
    parser.add_argument('--show', action='store_true', help='Show plots instead of saving')
    args = parser.parse_args()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    model_name = args.model if args.model else config.get("model", {}).get("name", "unet")
    if not model_name:
        raise ValueError("Model name not specified in config.yaml or command line")

    image_dir = config["data"]["processed"]["image"]["test"]
    mask_dir = config["data"]["processed"]["mask"]["test"]
    pred_dir = args.pred_dir if args.pred_dir else find_latest_pred_dir(model_name)
    output_dir = get_unique_dir(os.path.join("results", "comparison_plots", model_name))

    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory {image_dir} does not exist")
    if not os.path.exists(mask_dir):
        print(f"Warning: Mask directory {mask_dir} does not exist, skipping ground truth masks")
        mask_dir = None
    if not os.path.exists(pred_dir):
        print(f"Warning: Prediction directory {pred_dir} does not exist, skipping predictions")
        pred_dir = None

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        raise FileNotFoundError(f"No images found in {image_dir}")

    num_images = min(args.num_images, len(image_files))
    for img_name in image_files[:num_images]:
        image_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name) if mask_dir else None
        base_name = os.path.splitext(img_name)[0]
        pred_name = f"pred_{base_name}.jpg"
        pred_path = os.path.join(pred_dir, pred_name) if pred_dir else None

        print(f"Checking pred_path: {pred_path}")
        if mask_path and not os.path.exists(mask_path):
            print(f"Mask not found for {img_name}, skipping mask")
            mask_path = None
        if pred_path and not os.path.exists(pred_path):
            print(f"Prediction not found for {img_name}, skipping prediction")
            pred_path = None
        if not mask_path and not pred_path:
            print(f"Skipping {img_name}: No mask or prediction available")
            continue

        save_path = os.path.join(output_dir, f"vis_{img_name}") if not args.show else None
        visualize_prediction(image_path, mask_path, pred_path, save_path)
        if save_path:
            print(f"Saved visualization at {save_path}")

if __name__ == "__main__":
    main()