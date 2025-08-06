import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(torch.nn.Module):
    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        total = inputs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.eps) / (total + self.eps)
        return 1 - dice.mean()
class DiceBCELoss(nn.Module):
    def __init__(self, weight_bce=0.3, weight_dice=0.7):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, preds, targets):
        bce_loss = self.bce(preds, targets)
        pred_probs = torch.sigmoid(preds)
        smooth = 1e-6
        inter = (pred_probs * targets).sum((1,2,3))
        union = pred_probs.sum((1,2,3)) + targets.sum((1,2,3))
        dice_loss = 1 - ((2 * inter + smooth) / (union + smooth)).mean()
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        bce = F.binary_cross_entropy(preds, targets, reduction='none')
        pt = torch.where(targets == 1, preds, 1 - preds)
        loss = self.alpha * (1 - pt) ** self.gamma * bce
        return loss.mean()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1e-6):
        super(FocalDiceLoss, self).__init__()
        self.focal = FocalLoss(alpha, gamma)
        self.dice = DiceLoss(smooth)

    def forward(self, preds, targets):
        return self.focal(preds, targets) + self.dice(preds, targets)
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        TP = (inputs * targets).sum((1,2,3))
        FP = ((1 - targets) * inputs).sum((1,2,3))
        FN = (targets * (1 - inputs)).sum((1,2,3))
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        return (1 - Tversky).mean()

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, bce_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.tversky = TverskyLoss(alpha=alpha, beta=beta)

    def forward(self, pred, target):
        bce = F.binary_cross_entropy_with_logits(pred, target)
        tversky = self.tversky(pred, target)
        return self.bce_weight * bce + (1 - self.bce_weight) * tversky

class FocalTverskyLoss(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.3, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds = preds.view(-1)
        targets = targets.view(-1)

        TP = (preds * targets).sum()
        FP = ((1 - targets) * preds).sum()
        FN = (targets * (1 - preds)).sum()

        tversky = (TP + self.smooth) / (TP + self.alpha * FN + self.beta * FP + self.smooth)
        loss = (1 - tversky) ** self.gamma
        return loss
