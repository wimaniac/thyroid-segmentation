import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet34_Weights, ResNet50_Weights


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.5):
        super(ASPP, self).__init__()
        self.atrous_block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_block6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_block12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.atrous_block18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        size = x.shape[2:]
        x1 = self.atrous_block1(x)
        x2 = self.atrous_block6(x)
        x3 = self.atrous_block12(x)
        x4 = self.atrous_block18(x)
        x5 = self.global_avg_pool(x)
        x5 = nn.functional.interpolate(x5, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv1(x)
        return self.dropout(x)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, in_channels=3, backbone='resnet50', dropout=0.5, return_aux=False):
        super(DeepLabV3, self).__init__()
        self.return_aux = return_aux

        assert backbone in ['resnet18', 'resnet34', 'resnet50'], "Backbone must be 'resnet18', 'resnet34' or 'resnet50'"
        if backbone == 'resnet50':
            resnet = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if True else None)
            low_level_channels = 256
            aspp_in_channels = 2048
        elif backbone == 'resnet34':
            resnet = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if True else None)
            low_level_channels = 128
            aspp_in_channels = 512
        elif backbone == 'resnet18':
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if True else None)
            low_level_channels = 64
            aspp_in_channels = 512

        if in_channels != 3:
            self.initial_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                pretrained_conv1 = resnet.conv1.weight
                if in_channels == 1:
                    self.initial_conv.weight.copy_(pretrained_conv1.sum(dim=1, keepdim=True))
                else:
                    self.initial_conv.weight = nn.Parameter(pretrained_conv1[:, :in_channels])
        else:
            self.initial_conv = resnet.conv1

        self.backbone = nn.Sequential(
            self.initial_conv,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.aspp = ASPP(aspp_in_channels, 256, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_size = x.shape[2:]  # Lưu lại kích thước ảnh gốc

        features = []
        for i, layer in enumerate(self.backbone):
            x = layer(x)
            features.append(x)

        x = self.aspp(x)
        x = self.classifier(x)

        # Resize về đúng kích thước ảnh gốc
        x = nn.functional.interpolate(x, size=input_size, mode='bilinear', align_corners=False)
        return x