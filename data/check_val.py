import numpy as np
import os
from PIL import Image

def fg_ratio(mask_path):
    mask = np.array(Image.open(mask_path).convert("L"))
    fg = (mask > 127).sum()
    total = mask.size
    return fg / total

mask_dir = "data/processed/val/mask"
ratios = []
for fname in os.listdir(mask_dir):
    if fname.endswith('.jpg') or fname.endswith('.png'):
        ratios.append(fg_ratio(os.path.join(mask_dir, fname)))
ratios = np.array(ratios)
print("Val mask FG ratio stats: min", ratios.min(), "max", ratios.max(), "mean", ratios.mean(), "median", np.median(ratios))
# Liệt kê các file mask fg_ratio < 0.005
for fname, ratio in zip(os.listdir(mask_dir), ratios):
    if ratio < 0.002:
        print(f"{fname}: {ratio:.6f}")
