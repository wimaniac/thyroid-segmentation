import torch

def dice_score(pred, target, smooth=1e-6):
    """Tính Dice coefficient."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred, target, smooth=1e-6):
    """Tính IoU (Intersection over Union)."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(pred, target, smooth=1e-6):
    """Tính Precision."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    true_positive = (pred * target).sum()
    predicted_positive = pred.sum()
    return (true_positive + smooth) / (predicted_positive + smooth)

def recall_score(pred, target, smooth=1e-6):
    """Tính Recall."""
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    true_positive = (pred * target).sum()
    actual_positive = target.sum()
    return (true_positive + smooth) / (actual_positive + smooth)