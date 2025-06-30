import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.models.resnet import ResNet34_Weights


class UNetPlusPlus(nn.Module):
    def __init__(self, num_classes=1, dropout_rate=0.2):
        """
        UNet++ with ResNet34 backbone.

        Args:
            num_classes (int): Number of output classes (default: 1 for binary segmentation).
            dropout_rate (float): Dropout rate for regularization.
        """
        super(UNetPlusPlus, self).__init__()
        self.backbone = models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        # Modify initial convolution to accept single-channel input
        self.initial_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Encoder layers from ResNet34
        self.encoder1 = nn.Sequential(
            self.initial_conv,
            nn.BatchNorm2d(64),
            self.backbone.bn1,
            self.backbone.relu
        )
        self.encoder2 = self.backbone.layer1  # 64 channels
        self.encoder3 = self.backbone.layer2  # 128 channels
        self.encoder4 = self.backbone.layer3  # 256 channels
        self.encoder5 = self.backbone.layer4  # 512 channels

        # Pooling
        self.pool = nn.MaxPool2d(2, 2)

        # Nested decoder blocks
        self.decoder4_0 = self._conv_block(512, 256, dropout_rate)
        self.decoder3_0 = self._conv_block(256, 128, dropout_rate)
        self.decoder3_1 = self._conv_block(256 + 128, 128, dropout_rate)
        self.decoder2_0 = self._conv_block(128, 64, dropout_rate)
        self.decoder2_1 = self._conv_block(128 + 64, 64, dropout_rate)
        self.decoder2_2 = self._conv_block(128 + 64 + 64, 64, dropout_rate)
        self.decoder1_0 = self._conv_block(64, 32, dropout_rate)
        self.decoder1_1 = self._conv_block(64 + 32, 32, dropout_rate)
        self.decoder1_2 = self._conv_block(64 + 32 + 32, 32, dropout_rate)
        self.decoder1_3 = self._conv_block(64 + 32 + 32 + 32, 32, dropout_rate)

        # Upsampling layers to restore resolution
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_final = nn.Upsample(size=(256, 256), mode='bilinear', align_corners=True)

        # Final convolution
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

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
        # Encoder path
        x1 = self.initial_conv(x)  # Replace ResNet's conv1 for 1-channel input
        x1 = self.backbone.bn1(x1)
        x1 = self.backbone.relu(x1)  # 128x128
        x2 = self.pool(x1)  # 64x64
        x2 = self.encoder2(x2)  # 64 channels, 64x64
        x3 = self.encoder3(x2)  # 128 channels, 32x32
        x4 = self.encoder4(x3)  # 256 channels, 16x16
        x5 = self.encoder5(x4)  # 512 channels, 8x8

        # Decoder path (UNet++ nested structure)
        x4_0 = self.decoder4_0(x5)  # 256 channels, 8x8
        x4_0 = self.up(x4_0)  # 256 channels, 16x16

        x3_0 = self.decoder3_0(x4)  # 128 channels, 16x16
        x3_1 = self.decoder3_1(torch.cat([x3_0, x4_0], dim=1))  # 128 + 256 -> 128, 16x16
        x3_1 = self.up(x3_1)  # 128 channels, 32x32

        x2_0 = self.decoder2_0(x3)  # 64 channels, 32x32
        x2_1 = self.decoder2_1(torch.cat([x2_0, x3_1], dim=1))  # 64 + 128 -> 64, 32x32
        x2_2 = self.decoder2_2(torch.cat([x2_0, x2_1, x3_1], dim=1))  # 64 + 64 + 128 -> 64, 32x32
        x2_2 = self.up(x2_2)  # 64 channels, 64x64

        x1_0 = self.decoder1_0(x2)  # 32 channels, 64x64
        x1_1 = self.decoder1_1(torch.cat([x1_0, x2_2], dim=1))  # 32 + 64 -> 32, 64x64
        x1_2 = self.decoder1_2(torch.cat([x1_0, x1_1, x2_2], dim=1))  # 32 + 32 + 64 -> 32, 64x64
        x1_3 = self.decoder1_3(torch.cat([x1_0, x1_1, x1_2, x2_2], dim=1))  # 32 + 32 + 32 + 64 -> 32, 64x64

        # Final upsampling to 256x256
        x1_3 = self.up_final(x1_3)  # 32 channels, 256x256

        # Output
        out = self.final_conv(x1_3)  # 1 channel, 256x256
        return out


if __name__ == "__main__":
    model = UNetPlusPlus(num_classes=1)
    x = torch.randn(1, 1, 256, 256)
    output = model(x)
    print(f"Output shape: {output.shape}")