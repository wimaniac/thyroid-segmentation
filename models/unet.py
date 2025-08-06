import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class Unet(nn.Module):
    def __init__(self, dropout_rate, backbone="resnet34", in_channels=3, out_channels=1, pretrained=True):
        super(Unet, self).__init__()

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            encoder_channels = [64, 64, 128, 256, 512]  # Corrected
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone.fc = nn.Identity()

        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if pretrained:
            with torch.no_grad():
                pretrained_conv1 = self.backbone.conv1.weight
                if in_channels == 1:
                    self.initial_conv.weight.copy_(pretrained_conv1.sum(dim=1, keepdim=True))
                elif in_channels == 3:
                    self.initial_conv.weight = nn.Parameter(pretrained_conv1.clone())
                else:
                    raise ValueError(f"Unsupported in_channels: {in_channels}. Use 1 or 3.")

        self.enc1 = nn.Sequential(self.initial_conv, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.enc2 = self.backbone.layer1
        self.enc3 = self.backbone.layer2
        self.enc4 = self.backbone.layer3
        self.enc5 = self.backbone.layer4

        # Decoder
        self.up1 = self._upsample_block(encoder_channels[4], encoder_channels[3], dropout_rate)  # 512 -> 256
        self.up2 = self._upsample_block(encoder_channels[3], encoder_channels[2], dropout_rate)  # 256 -> 128
        self.up3 = self._upsample_block(encoder_channels[2], encoder_channels[1], dropout_rate)  # 128 -> 64
        self.up4 = self._upsample_block(encoder_channels[1], encoder_channels[0], dropout_rate)  # 64 -> 64

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[0], 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
    def _upsample_block(self, in_channels, out_channels, dropout):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x1 = self.enc1(x)  # 64x64
        x2 = self.enc2(x1)  # 64x64
        x3 = self.enc3(x2)  # 32x32
        x4 = self.enc4(x3)  # 16x16
        x5 = self.enc5(x4)  # 8x8

        d4 = self.up1(x5)  # 512 -> 256, 8x8 -> 16x16
        if d4.shape[2:] != x4.shape[2:]:
            x4 = F.interpolate(x4, size=d4.shape[2:], mode='bilinear', align_corners=False)
        d4 = d4 + x4

        d3 = self.up2(d4)  # 256 -> 128, 16x16 -> 32x32
        if d3.shape[2:] != x3.shape[2:]:
            x3 = F.interpolate(x3, size=d3.shape[2:], mode='bilinear', align_corners=False)
        d3 = d3 + x3

        d2 = self.up3(d3)  # 128 -> 64, 32x32 -> 64x64
        if d2.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=d2.shape[2:], mode='bilinear', align_corners=False)
        d2 = d2 + x2

        d1 = self.up4(d2)  # 64 -> 64, 64x64 -> 128x128
        if d1.shape[2:] != x1.shape[2:]:
            x1 = F.interpolate(x1, size=d1.shape[2:], mode='bilinear', align_corners=False)
        d1 = d1 + x1

        out = self.final_up(d1)  # 64 -> 64, 128x128 -> 256x256
        return self.final_conv(out)