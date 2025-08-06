import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myconfig1 import get_config
from data.thyroid_dataset import ThyroidDataset, get_test_transform
from metrics import dice_score, iou_score, precision_score, recall_score
from models.unet import Unet
from models.unetpp import UNetPlusPlus
from models.deeplabv3 import DeepLabV3

def load_model(model_path, model_type, device, in_channels=1, out_channels=1, backbone='resnet34', dropout_rate=0.5):
    if model_type == 'unet':
        model = Unet(dropout_rate=dropout_rate, backbone=backbone, in_channels=in_channels, out_channels=out_channels)
    elif model_type == 'unetpp':
        model = UNetPlusPlus(num_classes=out_channels, dropout_rate=dropout_rate, backbone=backbone, in_channels=in_channels)
    elif model_type == 'deeplabv3':
        model = DeepLabV3(num_classes=out_channels, in_channels=in_channels, backbone=backbone, dropout=dropout_rate)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def evaluate_model(model, test_loader, device):
    model.eval()
    dice_scores, iou_scores, prec_scores, rec_scores = [], [], [], []

    with torch.no_grad():
        for images, masks, _ in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            dice_scores.append(dice_score(outputs, masks))
            iou_scores.append(iou_score(outputs, masks))
            prec_scores.append(precision_score(outputs, masks))
            rec_scores.append(recall_score(outputs, masks))

    return {
        'dice': sum(dice_scores) / len(dice_scores),
        'iou': sum(iou_scores) / len(iou_scores),
        'precision': sum(prec_scores) / len(prec_scores),
        'recall': sum(rec_scores) / len(rec_scores)
    }

def main():
    args = get_config()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

    # Giả định mô hình đã được huấn luyện và lưu với checkpoint
    model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found at {model_path}. Please provide a valid path.")
        return

    # Tải dữ liệu test
    test_dataset = ThyroidDataset(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        transform=get_test_transform(),
        rgb= True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Tải mô hình
    model = load_model(model_path, args.model, device, in_channels=3, out_channels=1, backbone=args.backbone, dropout_rate=args.dropout_rate)

    # Đánh giá
    metrics = evaluate_model(model, test_loader, device)
    print(f"Evaluation Metrics on Test Set:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

if __name__ == '__main__':
    main()