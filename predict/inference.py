import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PIL import Image
import numpy as np
from models.unet import UNet
# from models.unetpp import UNetPP
# from models.deeplabv3 import DeepLabV3
from train.utils import load_checkpoint
from train.train_unet import ThyroidDataset

def get_model(model_name, in_channels=1, out_channels=1):
    """Khởi tạo mô hình dựa trên tên."""
    model_map = {
        'unet': UNet,
        # 'unetpp': UNetPP,
        # 'deeplabv3': DeepLabV3
    }
    if model_name not in model_map:
        raise ValueError(f"Model {model_name} not supported. Choose from {list(model_map.keys())}")
    return model_map[model_name](in_channels=in_channels, out_channels=out_channels)

def predict(model, loader, device, output_dir):
    """Dự đoán và lưu kết quả phân đoạn."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for idx, (data, _) in enumerate(loader):
            data = data.to(device)
            predictions = model(data)
            preds = torch.sigmoid(predictions) > 0.5
            preds = preds.cpu().numpy().squeeze()

            for i in range(preds.shape[0]):
                pred_img = (preds[i] * 255).astype(np.uint8)
                Image.fromarray(pred_img).save(os.path.join(output_dir, f"pred_{idx*loader.batch_size+i}.png"))

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Run inference with specified model")
    parser.add_argument('--model', type=str, default=None, help='Model name (unet, unetpp, deeplabv3)')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file')
    args = parser.parse_args()

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
    ])

    # Test dataset
    test_dataset = ThyroidDataset(
        image_dir=config["data"]["processed"]["image"]["test"],
        mask_dir=config["data"]["processed"]["mask"]["test"],
        transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)

    # Load model
    model_name = args.model if args.model else config["model"]["name"]
    checkpoint_path = args.checkpoint if args.checkpoint else os.path.join("checkpoints", model_name, f"best_{model_name}.pth")
    model = get_model(model_name, in_channels=1, out_channels=1).to(device)
    load_checkpoint(checkpoint_path, model)

    # Predict
    output_dir = os.path.join("results", "predictions", model_name)
    predict(model, test_loader, device, output_dir)
    print(f"Predictions saved to {output_dir}")

if __name__ == "__main__":
    main()