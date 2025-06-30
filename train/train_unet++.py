import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from models.unetpp import UNetPlusPlus
from train.utils import get_logger, save_checkpoint, EarlyStopping, DiceBCELoss, train_one_epoch, validate_one_epoch
from data.thyroid_dataset import ThyroidDataset, get_default_transform
from tqdm import tqdm
import argparse

def get_unique_log_file(base_path):
    """Tạo tên file log duy nhất bằng cách thêm số thứ tự nếu file đã tồn tại."""
    base, ext = os.path.splitext(base_path)
    counter = 1
    new_path = base_path
    while os.path.exists(new_path):
        new_path = f"{base}_{counter}{ext}"
        counter += 1
    return new_path

def get_model(model_name, in_channels=1, out_channels=1):
    model_map = {
        'unetpppretrained': UNetPlusPlus,
    }
    if model_name not in model_map:
        raise ValueError(f"Mô hình {model_name} không được hỗ trợ. Chọn từ {list(model_map.keys())}")
    return model_map[model_name](num_classes=out_channels)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình UNet++")
    parser.add_argument('--model', type=str, default='unetpppretrained', choices=['unetpppretrained'],
                        help="Loại mô hình để huấn luyện")
    args = parser.parse_args()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Cấu hình đã được nạp thành công")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = get_unique_log_file(os.path.join("logs", f"{args.model}.log"))
    logger = get_logger(log_file)

    # Data
    train_transform = get_default_transform(train=True)
    val_transform = get_default_transform(train=False)

    train_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["train"],
        mask_dir=config["data"]["processed"]["mask"]["train"],
        transform=train_transform,
        mixup_prob=0.3
    )
    val_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["val"],
        mask_dir=config["data"]["processed"]["mask"]["val"],
        transform=val_transform,
        mixup_prob=0.1

    )
    print(f"Kích thước tập huấn luyện: {len(train_dataset)}")
    print(f"Kích thước tập validation: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=4)

    # Model
    model = get_model(args.model, in_channels=1, out_channels=1).to(device)

    # Optimizer with weight decay
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=config["training"]["learning_rate"],
                           weight_decay=1e-3)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # Criterion
    criterion = DiceBCELoss()

    # Early stopping
    checkpoint_path = os.path.join("checkpoints", args.model, f"best_{args.model}.pth")
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        verbose=True,
        checkpoint_path=checkpoint_path,
        metric='dice+iou'
    )

    # Training loop
    # Training loop
    best_dice = 0
    checkpoint_dir = os.path.join("checkpoints", args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Bắt đầu huấn luyện {args.model.upper()}...")
    fine_tune_schedule = config["training"].get("fine_tune_schedule", {})
    for epoch in tqdm(range(config["training"]["epochs"]), desc="Epochs"):
        # Fine-tuning schedule
        if args.model == 'unetpppretrained' and (epoch + 1) in fine_tune_schedule:
            for name, param in model.named_parameters():
                if any(layer in name for layer in fine_tune_schedule[epoch + 1]):
                    param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config["training"]["learning_rate"] * 0.05,
                                   weight_decay=1e-4)

        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device,log_file)

        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # Update learning rate
        scheduler.step()

        # Early stopping based on combined Dice + IoU score
        combined_score = (val_dice + val_iou) / 2
        early_stopping(combined_score, model)
        if early_stopping.early_stop:
            logger.info("Dừng sớm được kích hoạt!")
            break

        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "best_dice": best_dice
            }, os.path.join(checkpoint_dir, f"epoch_{epoch + 1}.pth"))

    # Save final checkpoint
    save_checkpoint({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_dice": best_dice
    }, os.path.join(checkpoint_dir, f"final_{args.model}.pth"))

    logger.info(f"Hoàn thành huấn luyện! Mô hình tốt nhất được lưu tại {checkpoint_path}")

if __name__ == "__main__":
    main()