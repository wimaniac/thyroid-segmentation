import argparse
import torch
import yaml
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import dice_score, iou_score, precision_score, recall_score
from models.unet import UNet, PretrainedUNet
from models.unetpp import UNetPlusPlus
from train.utils import load_checkpoint
from data.thyroid_dataset import ThyroidDataset, get_default_transform

def accuracy_score(pred, target):
    """Tính accuracy cho phân đoạn."""
    preds = torch.sigmoid(pred) > 0.5  # Áp dụng ngưỡng 0.5
    correct = (preds == target).float().sum()  # Số pixel đúng
    total = target.numel()  # Tổng số pixel
    return correct / total

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

def evaluate_model(model, loader, device):
    """Đánh giá mô hình trên tập dữ liệu với metrics tùy chỉnh."""
    model.eval()
    total_dice = 0
    total_iou = 0
    total_precision = 0
    total_recall = 0
    total_accuracy = 0
    num_batches = len(loader)

    with torch.no_grad():
        for data, targets, _ in loader:
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            # Nếu mô hình trả về danh sách/tuple (deep supervision), lấy đầu ra cuối cùng
            if isinstance(predictions, (list, tuple)):
                predictions = predictions[-1]  # Chỉ sử dụng đầu ra chính

            preds = torch.sigmoid(predictions) > 0.5

            total_dice += dice_score(preds, targets).item()
            total_iou += iou_score(preds, targets).item()
            total_precision += precision_score(preds, targets).item()
            total_recall += recall_score(preds, targets).item()
            total_accuracy += accuracy_score(predictions, targets).item()

    return {
        'dice': total_dice / num_batches,
        'iou': total_iou / num_batches,
        'precision': total_precision / num_batches,
        'recall': total_recall / num_batches,
        'accuracy': total_accuracy / num_batches
    }

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Đánh giá mô hình trên tập dữ liệu test")
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
    print(f"Tải checkpoint từ {checkpoint_path}")

    # Evaluate
    results = evaluate_model(model, test_loader, device)
    print(f"{model_name.upper()} Test Results:")
    print(f"Dice: {results['dice']:.4f}")
    print(f"IoU: {results['iou']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")  # Thêm in accuracy

if __name__ == "__main__":
    main()