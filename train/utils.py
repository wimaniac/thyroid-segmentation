import torch
import torch.nn as nn
import logging
import os
import tqdm
from metrics import dice_score, iou_score
def accuracy_score(pred, target):
    """Tính accuracy cho phân đoạn."""
    preds = torch.sigmoid(pred) > 0.5  # Áp dụng ngưỡng 0.5
    correct = (preds == target).float().sum()  # Số pixel đúng
    total = target.numel()  # Tổng số pixel
    return correct / total
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, checkpoint_path="checkpoint.pth", metric='dice'):
        self.patience = patience
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.metric = metric

    def __call__(self, score, model):
        if self.metric == 'dice+iou':
            # Use combined score (Dice + IoU) / 2
            if self.best_score is None:
                self.best_score = score
                save_checkpoint(model.state_dict(), self.checkpoint_path)
            elif score > self.best_score:
                self.best_score = score
                save_checkpoint(model.state_dict(), self.checkpoint_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True
        else:
            # Original Dice-based early stopping
            if self.best_score is None:
                self.best_score = score
                save_checkpoint(model.state_dict(), self.checkpoint_path)
            elif score > self.best_score:
                self.best_score = score
                save_checkpoint(model.state_dict(), self.checkpoint_path)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose:
                    print(f"EarlyStopping counter: {self.counter}/{self.patience}")
                if self.counter >= self.patience:
                    self.early_stop = True

class DiceBCELoss(nn.Module):
    """Kết hợp BCE và Dice Loss với trọng số tùy chỉnh."""
    def __init__(self, weight_dice=0.7, weight_bce=0.3, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_bce = weight_bce
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        bce = self.bce_loss(pred, target)

        pred_flat = pred_sigmoid.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return self.weight_dice * dice_loss + self.weight_bce * bce

def get_logger(log_file):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding='utf-8')  # Thêm encoding='utf-8'
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def save_checkpoint(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
def load_checkpoint(model, optimizer=None, filename="checkpoint.pth"):
    """
    Tải checkpoint từ file đã lưu.

    Args:
        model (nn.Module): Mô hình để tải state_dict.
        optimizer (torch.optim.Optimizer, optional): Optimizer để tải state_dict nếu có.
        filename (str): Đường dẫn đến file checkpoint.

    Returns:
        dict: Thông tin checkpoint bao gồm epoch, best_dice (nếu có).
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Checkpoint file {filename} not found")

    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)

    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Trả về thông tin bổ sung (epoch, best_dice) nếu có
    info = {
        'epoch': checkpoint.get('epoch', 0),
        'best_dice': checkpoint.get('best_dice', 0.0)
    }
    return info

def train_one_epoch(model, loader, criterion, optimizer, device):
    """Huấn luyện một epoch với tqdm."""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    for data, targets, _ in tqdm.tqdm(loader, desc="Training", leave=False):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        predictions = model(data)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.sigmoid(predictions) > 0.5
        total_dice += dice_score(preds, targets).item()
        total_iou += iou_score(preds, targets).item()

    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)

def validate_one_epoch(model, loader, criterion, device, log_file):
    """Đánh giá trên tập validation với tqdm."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    logger = get_logger(log_file)  # Sử dụng log_file được truyền vào

    with torch.no_grad():
        for batch_idx, (data, targets, _) in enumerate(tqdm.tqdm(loader, desc="Validating", leave=False)):
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            preds = torch.sigmoid(predictions) > 0.5
            total_dice += dice_score(preds, targets).item()
            total_iou += iou_score(preds, targets).item()

            # Log batch gây dao động
            if loss.item() > 1.0:
                logger.warning(f"Batch {batch_idx} has high loss: {loss.item:.4f}, Data shape: {data.shape}")

    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)