import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from models.unet import UNet, PretrainedUNet
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
        'unet': UNet,
        'pretrained_unet': PretrainedUNet,
    }
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
    return model_map[model_name](in_channels=in_channels, out_channels=out_channels)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train UNet, PretrainedUNet, or TorchVisionUNet")
    parser.add_argument('--model', type=str, default='unet', choices=['unet', 'pretrained_unet', 'torch_unet'],
                        help="Model type to train")
    args = parser.parse_args()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("Config loaded successfully")

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_file = get_unique_log_file(os.path.join("logs", f"{args.model}.log"))
    logger = get_logger(log_file)

    # Data
    train_transform = get_default_transform()
    val_transform = get_default_transform()

    train_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["train"],
        mask_dir=config["data"]["processed"]["mask"]["train"],
        transform=train_transform,
    )
    val_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["val"],
        mask_dir=config["data"]["processed"]["mask"]["val"],
        transform=val_transform,
    )
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Model
    model = get_model(args.model, in_channels=1, out_channels=1).to(device)

    # Phase config (tối ưu model lần 2 tránh overfitting)
    phase_2_epoch = config["training"].get("unfreeze_epoch", 10)

    # Optimizer with weight decay
    weight_decay = 5e-4 if args.model == 'pretrained_unet' else 0
    if args.model == 'pretrained_unet':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=config["training"]["learning_rate"],
                               weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(),
                              lr=config["training"]["learning_rate"],
                              weight_decay=weight_decay)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    # Criterion
    criterion = DiceBCELoss()

    # Early stopping
    checkpoint_path = os.path.join("checkpoints", args.model, f"best_{args.model}.pth")
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        verbose=True,
        checkpoint_path=checkpoint_path
    )

    # Training loop
    best_dice = 0
    checkpoint_dir = os.path.join("checkpoints", args.model)
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Starting training {args.model.upper()}...")
    for epoch in tqdm(range(config["training"]["epochs"]), desc="Epochs"):
        # Phase 2: Unfreeze phần encoder
        if args.model == 'pretrained_unet' and epoch == phase_2_epoch:
            logger.info('Unfreezing encoder from layer4 for fine-tuning...')
            for name, param in model.named_parameters():
                if 'layer4' in name:
                    param.requires_grad = True
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                   lr=config["training"]["learning_rate"] * 0.1,
                                   weight_decay=1e-4)
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # Update learning rate
        scheduler.step(val_dice)

        # Early stopping based on Dice score
        early_stopping(val_dice, model)
        if early_stopping.early_stop:
            logger.info("Early stopping triggered!")
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

    logger.info(f"Training completed! Best model saved at {checkpoint_path}")

if __name__ == "__main__":
    main()