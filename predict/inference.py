import argparse
import torch
import yaml
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import numpy as np
from models.unet import UNet, PretrainedUNet
from train.utils import load_checkpoint
from data.thyroid_dataset import ThyroidDataset, get_default_transform
from models.unetpp import UNetPlusPlus


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


def get_model(model_name, in_channels=1, out_channels=1):
    """Khởi tạo mô hình dựa trên tên."""
    model_map = {
        'unet': UNet,
        'pretrained_unet': PretrainedUNet,
        'unetpppretrained': UNetPlusPlus,
    }
    if model_name not in model_map:
        raise ValueError(f"Mô hình {model_name} không được hỗ trợ. Chọn từ {list(model_map.keys())}")

    if model_name == 'unetpppretrained':
        # UNetPlusPlus sử dụng num_classes thay vì in_channels/out_channels
        return model_map[model_name](num_classes=out_channels)
    else:
        return model_map[model_name](in_channels=in_channels, out_channels=out_channels)


def predict(model, loader, device, output_dir):
    """Dự đoán và lưu kết quả phân đoạn."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for data, _, img_names in loader:  # Lấy img_names từ dataset
            data = data.to(device)
            predictions = model(data)
            preds = torch.sigmoid(predictions) > 0.5
            preds = preds.cpu().numpy().squeeze()

            for i, pred in enumerate(preds):
                pred_img = (pred * 255).astype(np.uint8)
                # Tạo tên file dự đoán dựa trên tên ảnh gốc
                base_name = os.path.splitext(img_names[i])[0]  # Lấy tên file không đuôi
                pred_path = get_unique_filename(os.path.join(output_dir, f"pred_{base_name}.jpg"))
                Image.fromarray(pred_img).save(pred_path)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Chạy dự đoán với mô hình được chỉ định")
    parser.add_argument('--model', type=str, default=None, help='Tên mô hình (unet, pretrained_unet, unetpppretrained)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Đường dẫn đến file checkpoint')
    args = parser.parse_args()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_default_transform(train=False)  # Sử dụng transform từ thyroid_dataset.py

    # Test dataset
    test_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["test"],
        mask_dir=config["data"]["processed"]["mask"]["test"],
        transform=transform,
        mask_suffix=".jpg"
    )
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    # Load model
    model_name = args.model if args.model else config["model"]["name"]
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join("checkpoints", model_name,
                                                                           f"best_{model_name}.pth")
    model = get_model(model_name, in_channels=1, out_channels=1).to(device)
    checkpoint_info = load_checkpoint(model, filename=checkpoint_path)
    print(
        f"Tải checkpoint từ {checkpoint_path}")

    # Predict
    output_dir = get_unique_dir(os.path.join("results", "predictions", model_name))
    predict(model, test_loader, device, output_dir)
    print(f"Dự đoán được lưu tại {output_dir}")


if __name__ == "__main__":
    main()