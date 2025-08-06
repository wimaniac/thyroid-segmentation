import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from datetime import datetime
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from data.thyroid_dataset import ThyroidDataset, get_train_transform, get_val_transform
from models.unetpp import UNetPlusPlus
from losses import FocalTverskyLoss
from myconfig1 import get_config
from metrics import dice_score, iou_score, precision_score, recall_score
from models.deeplabv3 import DeepLabV3

def get_log_filename(log_dir, base_name="training_log.csv"):
    os.makedirs(log_dir, exist_ok=True)
    base, ext = os.path.splitext(base_name)
    csv_file = os.path.join(log_dir, base_name)
    counter = 1
    while os.path.exists(csv_file):
        csv_file = os.path.join(log_dir, f"{base}_{counter}{ext}")
        counter += 1
    return csv_file


def save_log(args, metrics, epoch, log_filename):
    df = pd.DataFrame([metrics])
    mode = 'w' if epoch == 1 else 'a'
    header = epoch == 1
    df.to_csv(log_filename, mode=mode, header=header, index=False)


def save_checkpoint(model, epoch, args, filename='checkpoint.pth'):
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, filename)
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
        }, checkpoint_path)
    except PermissionError as e:
        print(f"PermissionError: Could not save checkpoint to {checkpoint_path}. Check directory permissions.")
        raise e


def visualize_predictions(images, masks, outputs, args, epoch, num_images=4):
    os.makedirs(os.path.join(args.save_dir, 'predictions'), exist_ok=True)
    fig, axes = plt.subplots(num_images, 3, figsize=(15, 5 * num_images))

    for i in range(min(num_images, len(images))):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = (img * 0.5 + 0.5).clip(0, 1)
        mask = masks[i].cpu().numpy().squeeze()
        pred = torch.sigmoid(outputs[i]).cpu().numpy().squeeze()
        pred = (pred > 0.5).astype(np.float32)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Image {i + 1}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'Ground Truth {i + 1}')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(pred, cmap='gray')
        axes[i, 2].set_title(f'Prediction {i + 1}')
        axes[i, 2].axis('off')

    plt.tight_layout()
    save_path = os.path.join(args.save_dir, 'predictions', f'epoch_{epoch + 1}_predictions.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    dice_scores, iou_scores, prec_scores, rec_scores = [], [], [], []

    for batch in tqdm(train_loader, desc='Training'):
        images, masks,_ = batch
        images, masks= images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)


        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        dice_scores.append(dice_score(outputs, masks))
        iou_scores.append(iou_score(outputs, masks))
        prec_scores.append(precision_score(outputs, masks))
        rec_scores.append(recall_score(outputs, masks))

    return {
        'train_loss': total_loss / len(train_loader),
        'train_dice': np.mean(dice_scores),
        'train_iou': np.mean(iou_scores),
        'train_precision': np.mean(prec_scores),
        'train_recall': np.mean(rec_scores)
    }


def validate(model, val_loader, criterion, device, args, epoch):
    model.eval()
    total_loss = 0
    dice_scores, iou_scores, prec_scores, rec_scores = [], [], [], []

    vis_images, vis_masks, vis_outputs = None, None, None
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader, desc='Validating')):
            images, masks, _ = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)



            total_loss += loss.item()

            dice_scores.append(dice_score(outputs, masks))
            iou_scores.append(iou_score(outputs, masks))
            prec_scores.append(precision_score(outputs, masks))
            rec_scores.append(recall_score(outputs, masks))

            if i == 0:
                vis_images, vis_masks, vis_outputs = images, masks, outputs

    if (epoch + 1) % 5 == 0 and vis_images is not None:
        visualize_predictions(vis_images, vis_masks, vis_outputs, args, epoch)

    return {
        'val_loss': total_loss / len(val_loader),
        'val_dice': np.mean(dice_scores),
        'val_iou': np.mean(iou_scores),
        'val_precision': np.mean(prec_scores),
        'val_recall': np.mean(rec_scores)
    }




def main():
    args = get_config()
    torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)

    try:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.join(args.save_dir, 'predictions'), exist_ok=True)
    except PermissionError as e:
        print(f"PermissionError: Could not create directories. Check permissions.")
        raise e

    log_filename = get_log_filename(args.log_dir)

    train_dataset = ThyroidDataset(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        transform=get_train_transform(),
        rgb=True
    )
    val_dataset = ThyroidDataset(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        transform=get_val_transform(),
        rgb=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )


    model = DeepLabV3(
        num_classes=1,
        dropout=args.dropout_rate,
        backbone=args.backbone,
        in_channels=3
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)


    criterion = FocalTverskyLoss(
        alpha=0.5,
        beta=0.5,
        gamma=1.3,
        smooth=1e-6
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,  # First cycle period
        T_mult=2,  # Multiplier for subsequent cycles
        eta_min=1e-6  # Minimum learning rate
    )

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=args.warmup_lr_init / args.learning_rate,
        total_iters=args.warmup_epochs
    )

    best_val_dice = 0
    patience_counter = 0
    start_epoch = 0

    for epoch in range(start_epoch, args.num_epochs):
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, args.device)

        val_metrics = validate(model, val_loader, criterion, args.device, args, epoch)

        metrics = {
            'epoch': epoch + 1,
            **train_metrics,
            **val_metrics
        }

        save_log(args, metrics, epoch + 1, log_filename)

        print(f'Epoch {epoch + 1}/{args.num_epochs}')
        print(
            f'Train Loss: {train_metrics["train_loss"]:.4f}, Dice: {train_metrics["train_dice"]:.4f}, IoU: {train_metrics["train_iou"]:.4f}')
        print(
            f'Val Loss: {val_metrics["val_loss"]:.4f}, Dice: {val_metrics["val_dice"]:.4f}, IoU: {val_metrics["val_iou"]:.4f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        if val_metrics['val_dice'] > best_val_dice:
            best_val_dice = val_metrics['val_dice']
            patience_counter = 0
            save_checkpoint(model, epoch, args, 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f'Early stopping at epoch {epoch + 1}')
                break


        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()


if __name__ == '__main__':
    main()