import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features[:-1]):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


class PretrainedUNet(nn.Module):
    def __init__(self, encoder_name="resnet34", in_channels=1, out_channels=1, pretrained=True, dropout_rate=0.5):
        super(PretrainedUNet, self).__init__()
        # Load pretrained encoder
        weights = 'IMAGENET1K_V1' if pretrained else None
        self.encoder = models.__dict__[encoder_name](weights=weights)
        self.encoder.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder.fc = nn.Identity()

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Encoder layers
        self.enc_layers = list(self.encoder.children())[:-2]
        self.enc_layers = nn.Sequential(*self.enc_layers)

        # Decoder with additional upsampling to reach 256x256
        self.upconv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),  # Từ 64x64 lên 128x128
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),  # 512 (256 + 256 từ skip) -> 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # Từ 128x128 lên 256x256
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.upconv3 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 256 (128 + 128 từ skip) -> 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # Từ 256x256 lên 512x512 (sẽ điều chỉnh sau)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate)
        )
        self.reduce_channels = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),  # Giảm từ 128 (64 + 64 từ skip) xuống 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv_final = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc_features = []
        for layer in self.enc_layers:
            x = layer(x)
            enc_features.append(x)

        # Decoder
        x = self.upconv1(enc_features[-1])  # Từ [16, 512, 64, 64] -> [16, 256, 128, 128]
        x = torch.cat([x, enc_features[-2]], dim=1)  # Concat với [16, 256, 128, 128] -> [16, 512, 128, 128]
        x = self.upconv2(x)  # Từ [16, 512, 128, 128] -> [16, 128, 256, 256]
        x = torch.cat([x, enc_features[-3]], dim=1)  # Concat với [16, 128, 256, 256] -> [16, 256, 256, 256]
        x = self.upconv3(x)  # Từ [16, 256, 256, 256] -> [16, 64, 512, 512]
        x = torch.cat([x, enc_features[-4]], dim=1)  # Concat với [16, 64, 256, 256] -> [16, 128, 512, 512]
        # Điều chỉnh kích thước về 256x256
        x = nn.functional.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        x = self.reduce_channels(x)  # Giảm từ 128 kênh xuống 64 kênh
        x = self.conv_final(x)

        return x


