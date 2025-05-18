import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from metrics import dice_score, iou_score


def get_logger(log_file):
    """Khởi tạo logger và lưu vào file."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(console_handler)

    return logger


def save_checkpoint(state, filename):
    """Lưu checkpoint mô hình."""
    torch.save(state, filename)
    print(f"Checkpoint saved: {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """Tải checkpoint mô hình."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    print(f"Checkpoint loaded: {checkpoint_path}")
    return checkpoint.get('epoch', 0), checkpoint.get('best_dice', 0)


class EarlyStopping:
    """Dừng sớm huấn luyện nếu không cải thiện sau số lần kiên nhẫn."""

    def __init__(self, patience=5, delta=0, verbose=False, checkpoint_path=None):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.checkpoint_path = checkpoint_path

    def __call__(self, val_score, model=None):
        if self.best_score is None:
            self.best_score = val_score
            if model and self.checkpoint_path:
                save_checkpoint({
                    "state_dict": model.state_dict(),
                    "epoch": 0,
                    "best_dice": val_score
                }, self.checkpoint_path)
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0
            if model and self.checkpoint_path:
                save_checkpoint({
                    "state_dict": model.state_dict(),
                    "epoch": 0,
                    "best_dice": val_score
                }, self.checkpoint_path)


class DiceBCELoss(nn.Module):
    """Kết hợp BCE và Dice Loss."""

    def __init__(self, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        pred_sigmoid = torch.sigmoid(pred)
        bce = self.bce_loss(pred, target)

        pred_flat = pred_sigmoid.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return bce + dice_loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Huấn luyện một epoch với tqdm."""
    model.train()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    # Thêm tqdm để hiển thị tiến trình
    for data, targets in tqdm(loader, desc="Training", leave=False):
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


def validate_one_epoch(model, loader, criterion, device):
    """Đánh giá trên tập validation với tqdm."""
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        # Thêm tqdm để hiển thị tiến trình
        for data, targets in tqdm(loader, desc="Validating", leave=False):
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            loss = criterion(predictions, targets)

            total_loss += loss.item()
            preds = torch.sigmoid(predictions) > 0.5
            total_dice += dice_score(preds, targets).item()
            total_iou += iou_score(preds, targets).item()

    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)