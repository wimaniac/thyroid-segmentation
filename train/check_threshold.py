import  sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from models.unet import PretrainedUNet
from myconfig1 import get_config
from data.thyroid_dataset import ThyroidDataset, get_default_transform
from torch.utils.data import DataLoader

def dice_coef(pred, gt, smooth=1e-5):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    num = 2 * (pred * gt).sum() + smooth
    den = pred.sum() + gt.sum() + smooth
    return num / den

def sweep_threshold(model, val_loader, device, thresholds=[0.3, 0.4, 0.5, 0.6]):
    model.eval()
    results = {th: [] for th in thresholds}
    img_names = []
    with torch.no_grad():
        for imgs, masks, names in tqdm(val_loader, desc="Sweep threshold"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            gts = masks.cpu().numpy()
            for i in range(imgs.shape[0]):
                gt = (gts[i,0] > 0.5).astype(np.uint8)
                for th in thresholds:
                    pred = (probs[i,0] > th).astype(np.uint8)
                    dice = dice_coef(pred, gt)
                    results[th].append(dice)
                img_names.append(names[i])
    # Tính dice trung bình mỗi threshold
    avg_dices = {th: np.mean(results[th]) for th in thresholds}
    print("Dice trung bình từng threshold:")
    for th, avg_d in avg_dices.items():
        print(f"Threshold={th:.2f}: Dice={avg_d:.4f}")
    return results, img_names

if __name__ == "__main__":
    args = get_config()

    val_ds = ThyroidDataset(args.val_image_dir, args.val_mask_dir, transform=get_default_transform(train=False), mask_suffix=args.mask_suffix, rgb=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = PretrainedUNet(encoder_name=args.backbone, in_channels=3, out_channels=1, pretrained=True, dropout_rate=args.dropout_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load checkpoint, model.to(device), .eval()
    thresholds = [0.2,0.25,0.3,0.35, 0.4,0.45, 0.5,0.55, 0.6]
    results, img_names = sweep_threshold(model, val_loader, device, thresholds)

    # Nếu muốn plot dice của từng ảnh ở từng threshold:
    import matplotlib.pyplot as plt
    dices_by_th = np.array([results[th] for th in thresholds])  # shape (n_th, n_img)
    plt.figure(figsize=(8,5))
    for i, th in enumerate(thresholds):
        plt.plot(dices_by_th[i], label=f"th={th}")
    plt.legend()
    plt.xlabel("Image idx")
    plt.ylabel("Dice score")
    plt.title("Dice từng ảnh theo threshold")
    plt.show()
