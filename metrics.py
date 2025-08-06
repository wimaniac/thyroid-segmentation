import torch


def dice_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    # Đảm bảo batch size giống nhau
    minibatch = min(pred.shape[0], target.shape[0])
    pred = pred[:minibatch]
    target = target[:minibatch]
    inter = (pred & target).float().sum((1,2,3))
    union = pred.float().sum((1,2,3)) + target.float().sum((1,2,3))
    return ((2 * inter + eps) / (union + eps)).mean().item()

def iou_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    # Đảm bảo batch size giống nhau
    minibatch = min(pred.shape[0], target.shape[0])
    pred = pred[:minibatch]
    target = target[:minibatch]
    inter = (pred & target).float().sum((1,2,3))
    union = (pred | target).float().sum((1,2,3))
    return ((inter + eps) / (union + eps)).mean().item()

def precision_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    minibatch = min(pred.shape[0], target.shape[0])
    pred = pred[:minibatch]
    target = target[:minibatch]
    true_positives = (pred & target).float().sum((1, 2, 3))
    predicted_positives = pred.float().sum((1, 2, 3))
    return ((true_positives + eps) / (predicted_positives + eps)).mean().item()

def recall_score(pred, target, eps=1e-7):
    pred = torch.sigmoid(pred) > 0.5
    target = target > 0.5
    minibatch = min(pred.shape[0], target.shape[0])
    pred = pred[:minibatch]
    target = target[:minibatch]
    true_positives = (pred & target).float().sum((1, 2, 3))
    actual_positives = target.float().sum((1, 2, 3))
    return ((true_positives + eps) / (actual_positives + eps)).mean().item()
