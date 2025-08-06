import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from myconfig1 import get_config
from data.thyroid_dataset import ThyroidDataset, get_test_transform
from models.unet import Unet
from models.unetpp import UNetPlusPlus
from models.deeplabv3 import DeepLabV3
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral
def load_model(model_path, model_type, device, in_channels=3, out_channels=1, backbone='resnet18', dropout_rate=0.5):
    if model_type == 'unet':
        model = Unet(dropout_rate=dropout_rate, backbone=backbone, in_channels=in_channels, out_channels=out_channels)
    elif model_type == 'unetpp':
        model = UNetPlusPlus(num_classes=out_channels, dropout_rate=dropout_rate, backbone=backbone, in_channels=in_channels)
    elif model_type == 'deeplabv3':
        model = DeepLabV3(num_classes=out_channels, in_channels=in_channels, backbone=backbone, dropout=dropout_rate)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict['model_state_dict'])
    model.to(device)
    model.eval()
    return model

def smooth_mask(pred_mask, sigma=2):
    blurred = cv2.GaussianBlur(pred_mask.astype(np.float32), (0, 0), sigma)
    smoothed_mask = (blurred > 127).astype(np.uint8) * 255
    return smoothed_mask

def apply_crf(image, prob, t=5, sxy=80, srgb=13):
    n_classes = 2
    d = dcrf.DenseCRF2D(image.shape[1], image.shape[0], n_classes)
    U = unary_from_softmax(prob)
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=sxy, srgb=srgb, rgbim=image, compat=10)
    Q = d.inference(t)
    return np.argmax(Q, axis=0).reshape(image.shape[:2]).astype(np.uint8) * 255

def ensemble_predict(models, image, device):
    probs = []
    for model in models:
        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(image)).cpu().numpy()
            probs.append(prob)
    avg_prob = np.mean(probs, axis=0)
    return (avg_prob > 0.5).astype(np.uint8) * 255

def normalize_mask(pred_mask):
    pred_mask = np.clip(pred_mask, 0, 255)
    pred_mask = (pred_mask > 127).astype(np.uint8) * 255
    return pred_mask

def fill_holes(pred_mask, kernel_size=5):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.morphologyEx(pred_mask, cv2.MORPH_CLOSE, kernel)

def remove_small_regions(pred_mask, min_area=100):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(pred_mask, connectivity=8)
    for label in range(1, len(stats)):
        if stats[label, cv2.CC_STAT_AREA] < min_area:
            pred_mask[labels == label] = 0
    return pred_mask
def infer_model(model, test_loader, device, pred_dir=None):
    model.eval()
    os.makedirs(pred_dir, exist_ok=True) if pred_dir else None

    with torch.no_grad():
        for i, (images, _, image_paths) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            for j in range(outputs.size(0)):
                pred_mask = outputs[j].cpu().numpy().squeeze()

                # Áp dụng hậu xử lý
                pred_mask = (pred_mask > 0.5).astype(np.float32) * 255  # Nhị phân hóa
                pred_mask = smooth_mask(pred_mask, sigma=2)  # Làm mịn
                pred_mask = remove_small_regions(pred_mask, min_area=100)  # Loại bỏ nhiễu
                pred_mask = fill_holes(pred_mask, kernel_size=5)  # Điền lỗ
                pred_mask = normalize_mask(pred_mask)  # Chuẩn hóa

                if pred_dir:
                    filename = os.path.basename(image_paths[j])
                    mask_path = os.path.join(pred_dir, f"{filename}")
                    cv2.imwrite(mask_path, pred_mask)

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
        mask_dir=args.test_mask_dir,  # Không cần mask để inference, nhưng giữ để tương thích
        transform=get_test_transform(),
        rgb=True  # Đảm bảo xử lý ảnh 3 kênh
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

    # Thực hiện inference
    infer_model(model, test_loader, device, pred_dir=args.pred_dir)
    print(f"Inference completed. Predictions saved to {args.pred_dir} if specified.")

if __name__ == '__main__':
    main()