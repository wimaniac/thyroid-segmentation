import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights

class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.5, backbone='resnet50', in_channels=1):
        super(UNetPlusPlus, self).__init__()
        self.in_channels = in_channels

        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            encoder_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if in_channels == 1:
            with torch.no_grad():
                pretrained_conv1 = self.backbone.conv1.weight
                self.initial_conv.weight.copy_(pretrained_conv1.sum(dim=1, keepdim=True))
        elif in_channels == 3:
            self.initial_conv.weight = nn.Parameter(self.backbone.conv1.weight.clone())

        self.encoder1 = nn.Sequential(
            self.initial_conv,
            nn.BatchNorm2d(64),
            self.backbone.bn1,
            self.backbone.relu
        )
        self.encoder2 = self.backbone.layer1
        self.encoder3 = self.backbone.layer2
        self.encoder4 = self.backbone.layer3
        self.encoder5 = self.backbone.layer4

        self.pool = nn.MaxPool2d(2, 2)

        self.decoder4_0 = self._conv_block(encoder_channels[4], 1024, dropout_rate)
        self.decoder3_0 = self._conv_block(encoder_channels[3], 512, dropout_rate)
        self.decoder3_1 = self._conv_block(512 + 1024, 512, dropout_rate)
        self.decoder2_0 = self._conv_block(encoder_channels[2], 256, dropout_rate)
        self.decoder2_1 = self._conv_block(256 + 512, 256, dropout_rate)
        self.decoder2_2 = self._conv_block(256 + 256 + 512, 256, dropout_rate)
        self.decoder1_0 = self._conv_block(encoder_channels[1], 128, dropout_rate)
        self.decoder1_1 = self._conv_block(128 + 256, 128, dropout_rate)
        self.decoder1_2 = self._conv_block(128 + 128 + 256, 128, dropout_rate)
        self.decoder1_3 = self._conv_block(128 + 128 + 128 + 256, 128, dropout_rate)

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_final = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)
        self.final_conv = nn.Conv2d(128, num_classes, kernel_size=1)

    def _conv_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate)
        )

    def forward(self, x):
        x1 = self.initial_conv(x)
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)
        x2 = self.pool(x1)
        x2 = self.encoder2(x2)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)
        x5 = self.encoder5(x4)

        x4_0 = self.decoder4_0(x5)
        x4_0 = self.up(x4_0)

        x3_0 = self.decoder3_0(x4)
        x3_1 = self.decoder3_1(torch.cat([x3_0, x4_0], dim=1))
        x3_1 = self.up(x3_1)

        x2_0 = self.decoder2_0(x3)
        x2_1 = self.decoder2_1(torch.cat([x2_0, x3_1], dim=1))
        x2_2 = self.decoder2_2(torch.cat([x2_0, x2_1, x3_1], dim=1))
        x2_2 = self.up(x2_2)

        x1_0 = self.decoder1_0(x2)
        x1_1 = self.decoder1_1(torch.cat([x1_0, x2_2], dim=1))
        x1_2 = self.decoder1_2(torch.cat([x1_0, x1_1, x2_2], dim=1))
        x1_3 = self.decoder1_3(torch.cat([x1_0, x1_1, x1_2, x2_2], dim=1))
        x1_3 = self.up_final(x1_3)

        out_main = self.final_conv(x1_3)
        return out_main