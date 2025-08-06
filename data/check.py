if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from tqdm import tqdm
    from torch.utils.data import DataLoader
    from myconfig1 import get_config
    from models.unet import PretrainedUNet
    from data.thyroid_dataset import ThyroidDataset, get_default_transform

    def dice_coef(pred, gt, smooth=1e-5):
        pred = pred.astype(np.float32)
        gt = gt.astype(np.float32)
        num = 2 * (pred * gt).sum() + smooth
        den = pred.sum() + gt.sum() + smooth
        return num / den

    args = get_config()

    model = PretrainedUNet(encoder_name=args.backbone, in_channels=3, out_channels=1, pretrained=True, dropout_rate=args.dropout_rate)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('checkpoints/unet/best_model.pth', map_location=device))
    model = model.to(device)
    model.eval()

    val_ds = ThyroidDataset(args.val_image_dir, args.val_mask_dir, transform=get_default_transform(train=False),
                            mask_suffix=args.mask_suffix, rgb=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    n_show = 30
    count = 0

    with torch.no_grad():
        for imgs, masks, names in tqdm(val_loader, desc="Visualizing val"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            gts = masks.cpu().numpy()
            imgs_np = imgs.cpu().numpy()
            for i in range(imgs.shape[0]):
                pred = (probs[i, 0] > 0.5).astype(np.uint8)
                gt = (gts[i, 0] > 0.5).astype(np.uint8)
                # Sửa shape nếu cần
                if gt.shape != pred.shape:
                    from skimage.transform import resize

                    gt = resize(gt, pred.shape, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint8)
                dice = dice_coef(pred, gt)
                pred = (probs[i,0] > 0.5).astype(np.uint8)
                gt = (gts[i,0] > 0.5).astype(np.uint8)
                img_show = imgs_np[i].transpose(1,2,0)
                if img_show.max() <= 1.0:
                    img_show = (img_show * 255).astype(np.uint8)
                fig, axes = plt.subplots(1, 3, figsize=(12,4))
                axes[0].imshow(img_show, cmap='gray')
                axes[0].set_title(f"Input ({names[i]})")
                axes[1].imshow(gt, cmap='gray')
                axes[1].set_title("Ground Truth")
                axes[2].imshow(pred, cmap='gray')
                axes[2].set_title(f"Pred (Dice={dice:.3f})")
                for ax in axes: ax.axis("off")
                plt.tight_layout()
                plt.show()
                count += 1
                if count >= n_show:
                    break
            if count >= n_show:
                break
