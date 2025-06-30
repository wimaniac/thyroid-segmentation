import argparse
import os
import re
import matplotlib.pyplot as plt
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Thêm đường dẫn đến thư mục cha
from models.unet import UNet, PretrainedUNet
# from models.deeplabv3 import DeepLabV3
from train.utils import load_checkpoint
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
        # 'deeplabv3': DeepLabV3
    }
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
    return model_map[model_name](in_channels=in_channels, out_channels=out_channels)

def parse_log_file(log_file):
    """Trích xuất metrics từ file log nếu không có CSV."""
    epochs = []
    train_losses, train_dices, train_ious = [], [], []
    val_losses, val_dices, val_ious = [], [], []

    with open(log_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Train Loss" in line:
                match = re.match(r".*Train Loss: ([\d.]+), Dice: ([\d.]+), IoU: ([\d.]+)", line)
                if match:
                    train_losses.append(float(match.group(1)))
                    train_dices.append(float(match.group(2)))
                    train_ious.append(float(match.group(3)))
            if "Val Loss" in line:
                match = re.match(r".*Val Loss: ([\d.]+), Dice: ([\d.]+), IoU: ([\d.]+)", line)
                if match:
                    val_losses.append(float(match.group(1)))
                    val_dices.append(float(match.group(2)))
                    val_ious.append(float(match.group(3)))
            if "Epoch" in line:
                match = re.match(r".*Epoch (\d+)/\d+", line)
                if match:
                    epochs.append(int(match.group(1)))

    return {
        'epochs': epochs,
        'train_loss': train_losses,
        'train_dice': train_dices,
        'train_iou': train_ious,
        'val_loss': val_losses,
        'val_dice': val_dices,
        'val_iou': val_ious
    }

def plot_metrics(metrics, model_name, save_dir):
    """Vẽ và lưu biểu đồ."""
    # Tạo thư mục con theo tên mô hình với hậu tố nếu cần
    model_save_dir = get_unique_dir(os.path.join(save_dir, model_name))
    os.makedirs(model_save_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epochs'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epochs'], metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name.upper()} Training Loss')
    plt.legend()
    plt.grid(True)
    loss_path = get_unique_filename(os.path.join(model_save_dir, f'{model_name}_loss.png'))
    plt.savefig(loss_path)
    plt.close()

    # Plot Dice
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epochs'], metrics['train_dice'], label='Train Dice')
    plt.plot(metrics['epochs'], metrics['val_dice'], label='Validation Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.title(f'{model_name.upper()} Dice Score')
    plt.legend()
    plt.grid(True)
    dice_path = get_unique_filename(os.path.join(model_save_dir, f'{model_name}_dice.png'))
    plt.savefig(dice_path)
    plt.close()

    # Plot IoU
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epochs'], metrics['train_iou'], label='Train IoU')
    plt.plot(metrics['epochs'], metrics['val_iou'], label='Validation IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.title(f'{model_name.upper()} IoU Score')
    plt.legend()
    plt.grid(True)
    iou_path = get_unique_filename(os.path.join(model_save_dir, f'{model_name}_iou.png'))
    plt.savefig(iou_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument('--model', type=str, default='unet', help='Model name (unet, pretrained_unet, unetpppretrained)')
    parser.add_argument('--log', type=str, default=None, help='Path to log file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file (optional for model validation)')
    args = parser.parse_args()

    model_name = args.model
    save_dir = os.path.join("results", "training_plots")

    if args.log and os.path.exists(args.log):
        metrics = parse_log_file(args.log)
    else:
        log_file = os.path.join("logs", f"{model_name}.log")
        if os.path.exists(log_file):
            metrics = parse_log_file(log_file)
        else:
            raise FileNotFoundError(f"No metrics file found for {model_name}")

    plot_metrics(metrics, model_name, save_dir)
    print(f"Training plots saved to {get_unique_dir(os.path.join(save_dir, model_name))}")

    # Optional: Load and validate model if checkpoint is provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = get_model(model_name, in_channels=1, out_channels=1).to(device)
        load_checkpoint(args.checkpoint, model)
        print(f"Model {model_name} loaded from checkpoint for validation.")

if __name__ == "__main__":
    main()