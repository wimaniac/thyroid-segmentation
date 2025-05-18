import argparse
import os
import csv
import re
import matplotlib.pyplot as plt


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


def read_metrics_csv(csv_file):
    """Đọc metrics từ file CSV."""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'val_loss': [],
        'val_dice': [],
        'val_iou': []
    }
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics['epochs'].append(int(row['epoch']))
            metrics['train_loss'].append(float(row['train_loss']))
            metrics['train_dice'].append(float(row['train_dice']))
            metrics['train_iou'].append(float(row['train_iou']))
            metrics['val_loss'].append(float(row['val_loss']))
            metrics['val_dice'].append(float(row['val_dice']))
            metrics['val_iou'].append(float(row['val_iou']))
    return metrics


def plot_metrics(metrics, model_name, save_dir):
    """Vẽ và lưu biểu đồ."""
    os.makedirs(save_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['epochs'], metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['epochs'], metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{model_name.upper()} Training Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'{model_name}_loss.png'))
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
    plt.savefig(os.path.join(save_dir, f'{model_name}_dice.png'))
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
    plt.savefig(os.path.join(save_dir, f'{model_name}_iou.png'))
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument('--model', type=str, default='unet', help='Model name (unet, unetpp, deeplabv3)')
    parser.add_argument('--csv', type=str, default=None, help='Path to metrics CSV file')
    parser.add_argument('--log', type=str, default=None, help='Path to log file')
    args = parser.parse_args()

    model_name = args.model
    save_dir = os.path.join("results", "training_plots")

    # Ưu tiên CSV nếu có
    if args.csv and os.path.exists(args.csv):
        metrics = read_metrics_csv(args.csv)
    elif args.log and os.path.exists(args.log):
        metrics = parse_log_file(args.log)
    else:
        csv_file = os.path.join("logs", f"{model_name}_metrics.csv")
        log_file = os.path.join("logs", f"{model_name}.log")
        if os.path.exists(csv_file):
            metrics = read_metrics_csv(csv_file)
        elif os.path.exists(log_file):
            metrics = parse_log_file(log_file)
        else:
            raise FileNotFoundError(f"No metrics file found for {model_name}")

    plot_metrics(metrics, model_name, save_dir)
    print(f"Training plots saved to {save_dir}")


if __name__ == "__main__":
    main()