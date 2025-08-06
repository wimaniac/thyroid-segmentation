import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import matplotlib.pyplot as plt
from myconfig1 import get_config
def visualize_training_log(csv_file, save_dir):
    # Đọc file CSV
    df = pd.read_csv(csv_file)

    # Kiểm tra dữ liệu và lọc bỏ hàng NaN nếu có
    df = df.dropna()
    df = df.sort_values("epoch")
    df = df.drop_duplicates(subset="epoch", keep="last")

    # Tìm epoch tốt nhất dựa trên val_dice
    best_idx = df['val_dice'].idxmax()
    best_epoch = int(df.loc[best_idx, 'epoch'])
    best_dice = df.loc[best_idx, 'val_dice']

    # Danh sách metric để trực quan hóa
    metrics = [
        ("Dice", "train_dice", "val_dice"),
        ("IoU", "train_iou", "val_iou"),
        ("Loss", "train_loss", "val_loss"),
        ("Precision", "train_precision", "val_precision"),
        ("Recall", "train_recall", "val_recall"),
    ]

    # Tạo thư mục lưu nếu chưa tồn tại
    os.makedirs(save_dir, exist_ok=True)

    # Trực quan hóa từng metric
    for metric_name, train_col, val_col in metrics:
        plt.figure(figsize=(8, 5))

        if train_col in df.columns:
            plt.plot(df["epoch"], df[train_col], label=f"Train {metric_name}")
        if val_col in df.columns:
            plt.plot(df["epoch"], df[val_col], label=f"Val {metric_name}")

        if metric_name == "Dice":
            plt.scatter(best_epoch, best_dice, c='red', label=f'Best Dice: {best_dice:.4f}')
            plt.annotate(f'{best_dice:.4f}', (best_epoch, best_dice),
                         textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10, color='red')

        plt.xlabel("Epoch")
        plt.ylabel(metric_name)
        plt.title(f"{metric_name} over Epochs")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Lưu hình ảnh
        plt.savefig(os.path.join(save_dir, f"{metric_name.lower()}_curve.png"))
        plt.close()

if __name__ == "__main__":
    args = get_config()

    visualize_training_log(args.csv_file, args.save_dir)
    print(f"Visualization completed. Images saved to {args.save_dir}.")