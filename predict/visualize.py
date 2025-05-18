import os
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def visualize_prediction(image_path, mask_path=None, pred_path=None, save_path=None):
    """Hiển thị hoặc lưu ảnh gốc, mask thật, và mask dự đoán."""
    image = np.array(Image.open(image_path).convert("L"))
    mask = np.array(Image.open(mask_path).convert("L")) if mask_path else None
    pred = np.array(Image.open(pred_path).convert("L")) if pred_path else None

    fig, axes = plt.subplots(1, 3 if mask is not None else 2, figsize=(12, 4))

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
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    image_dir = config["data"]["processed"]["image"]["test"]
    mask_dir = config["data"]["processed"]["mask"]["test"]
    pred_dir = os.path.join("results", "predictions", config["model"]["name"])
    output_dir = os.path.join("results", "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for img_name in image_files[:5]:  # Visualize first 5 images
        image_path = os.path.join(image_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        pred_name = f"pred_{image_files.index(img_name)}.png"
        pred_path = os.path.join(pred_dir, pred_name)

        if not os.path.exists(mask_path) or not os.path.exists(pred_path):
            print(f"Skipping {img_name}: Mask or prediction not found")
            continue

        save_path = os.path.join(output_dir, f"vis_{img_name}")
        visualize_prediction(image_path, mask_path, pred_path, save_path)
        print(f"Saved visualization at {save_path}")

if __name__ == "__main__":
    main()