import sys
import os
# Thêm thư mục gốc vào sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import yaml
import numpy as np
from PIL import Image
from models.unet import UNet
from train.utils import get_logger, save_checkpoint, EarlyStopping, DiceBCELoss, train_one_epoch, validate_one_epoch
from tqdm import tqdm

class ThyroidDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask = (mask > 0).astype(np.float32)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def main():
    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(os.path.join("logs", "unet.log"))

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["train"],
        mask_dir=config["data"]["processed"]["mask"]["train"],
        transform=transform
    )
    val_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["val"],
        mask_dir=config["data"]["processed"]["mask"]["val"],
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Model
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = DiceBCELoss()

    # Early stopping
    checkpoint_path = os.path.join("checkpoints", "unet", "best_unet.pth")
    early_stopping = EarlyStopping(
        patience=config["training"]["patience"],
        verbose=True,
        checkpoint_path=checkpoint_path
    )

    # Training loop
    best_dice = 0
    checkpoint_dir = os.path.join("checkpoints", "unet")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info("Starting training U-Net...")
    for epoch in tqdm(range(config["training"]["epochs"]), desc="Epochs"):
        train_loss, train_dice, train_iou = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        logger.info(f"Epoch {epoch + 1}/{config['training']['epochs']}")
        logger.info(f"Train Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")

        # Early stopping based on Dice score (higher is better)
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
    }, os.path.join(checkpoint_dir, "final_unet.pth"))

    logger.info(f"Training completed! Best model saved at {checkpoint_path}")

if __name__ == "__main__":
    main()