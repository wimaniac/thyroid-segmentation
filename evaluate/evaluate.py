import argparse
import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from metrics import dice_score, iou_score, precision_score, recall_score
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

def evaluate_model(model, loader, device):
   """Đánh giá mô hình trên tập dữ liệu với metrics tùy chỉnh."""
   model.eval()
   total_dice = 0
   total_iou = 0
   total_precision = 0
   total_recall = 0
   num_batches = len(loader)

   with torch.no_grad():
       for data, targets in loader:
           data = data.to(device)
           targets = targets.to(device)

           predictions = model(data)
           preds = torch.sigmoid(predictions) > 0.5

           total_dice += dice_score(preds, targets).item()
           total_iou += iou_score(preds, targets).item()
           total_precision += precision_score(preds, targets).item()
           total_recall += recall_score(preds, targets).item()

   return {
       'dice': total_dice / num_batches,
       'iou': total_iou / num_batches,
       'precision': total_precision / num_batches,
       'recall': total_recall / num_batches
   }

def main():
   # Parse arguments
   parser = argparse.ArgumentParser(description="Evaluate model on test dataset")
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

   # Evaluate
   results = evaluate_model(model, test_loader, device)
   print(f"{model_name.upper()} Test Results:")
   print(f"Dice: {results['dice']:.4f}")
   print(f"IoU: {results['iou']:.4f}")
   print(f"Precision: {results['precision']:.4f}")
   print(f"Recall: {results['recall']:.4f}")

if __name__ == "__main__":
   main()