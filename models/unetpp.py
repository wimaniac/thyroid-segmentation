import torch
import torch.nn as nn


class UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=[64, 128, 256, 512], deep_supervision=True):
        super(UNetPlusPlus, self).__init__()
        self.deep_supervision = deep_supervision
        self.features = features
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self.conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self.conv_block(features[-1], features[-1] * 2)

        # Decoder (nested nodes X^{i,j})
        self.decoder = nn.ModuleDict()
        for i in range(len(features)):  # Tầng (0, 1, 2, 3)
            for j in range(1, len(features) - i):  # Node trong tầng (j=1,2,3)
                in_ch = features[i] * (j + 1)  # Số kênh đầu vào = features[i] * (1 từ upsample + j từ skip)
                self.decoder[f"X_{i}_{j}"] = self.conv_block(in_ch, features[i])
                if i < len(features) - 1:  # Không cần upsample cho tầng cuối
                    self.decoder[f"up_{i}_{j}"] = nn.ConvTranspose2d(features[i], features[i], kernel_size=2, stride=2)

        # Deep supervision outputs
        self.supervision_heads = nn.ModuleList()
        if deep_supervision:
            for j in range(1, len(features)):
                self.supervision_heads.append(nn.Conv2d(features[0], out_channels, kernel_size=1))

        # Final output (X^{0,0})
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc_features = []
        for enc in self.encoder:
            x = enc(x)
            enc_features.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        enc_features.append(x)

        # Decoder (nested nodes)
        nodes = {}
        for i in range(len(self.features) - 1, -1, -1):  # Tầng: 3, 2, 1, 0
            for j in range(len(self.features) - i):  # Node: 0, 1, 2, 3
                if j == 0:  # Node X^{i,0} (encoder)
                    nodes[f"X_{i}_0"] = enc_features[i]
                else:
                    # Concatenate inputs
                    inputs = [nodes[f"X_{i}_{j - 1}"]]  # Từ node trước trong cùng tầng
                    if i < len(self.features) - 1:  # Từ node dưới (upsample)
                        up_key = f"up_{i + 1}_{j - 1}"
                        upsampled = self.decoder[up_key](nodes[f"X_{i + 1}_{j - 1}"])
                        inputs.append(upsampled)
                    x = torch.cat(inputs, dim=1)
                    # Apply conv block
                    x = self.decoder[f"X_{i}_{j}"](x)
                    nodes[f"X_{i}_{j}"] = x

        # Deep supervision outputs
        outputs = []
        if self.deep_supervision:
            for j in range(1, len(self.features)):
                out = self.supervision_heads[j - 1](nodes[f"X_0_{j}"])
                out = nn.functional.interpolate(out, size=nodes[f"X_0_0"].shape[2:], mode='bilinear',
                                                align_corners=False)
                outputs.append(out)

        # Final output
        final_out = self.final_conv(nodes[f"X_0_0"])
        outputs.append(final_out)

        # Return outputs
        if self.deep_supervision and self.training:
            return [torch.sigmoid(out) for out in outputs]
        return torch.sigmoid(final_out)



