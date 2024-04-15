import torch
import torch.nn as nn


# Define MBConvBlock
class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio=1):
        super(MBConvBlock, self).__init__()
        self.expand_conv = nn.Conv2d(
            in_channels, in_channels * expand_ratio, kernel_size=1, bias=False
        )
        self.depthwise_conv = nn.Conv2d(
            in_channels * expand_ratio,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=1,
            groups=in_channels * expand_ratio,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.se_reduce = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1)
        self.se_expand = nn.Conv2d(out_channels // 4, out_channels, kernel_size=1)
        self.project_conv = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.expand_conv(x)
        x = self.relu(x)
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        # SE block
        se = torch.mean(x, dim=(-2, -1), keepdim=True)
        se = self.se_expand(self.relu(self.se_reduce(se)))
        x = x * torch.sigmoid(se)
        x = self.project_conv(x)
        x = self.bn2(x)
        # Skip connection
        if x.shape == identity.shape:
            x += identity
        return x


# Define Encoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_stem = nn.Conv2d(
            3, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.bn0 = nn.BatchNorm2d(32)
        self.blocks = nn.ModuleList(
            [
                MBConvBlock(32, 16, kernel_size=3, stride=1, expand_ratio=1),
                MBConvBlock(16, 24, kernel_size=3, stride=2, expand_ratio=6),
                MBConvBlock(24, 24, kernel_size=3, stride=1, expand_ratio=6),
                MBConvBlock(24, 40, kernel_size=5, stride=2, expand_ratio=6),
                MBConvBlock(40, 40, kernel_size=5, stride=1, expand_ratio=6),
                MBConvBlock(40, 80, kernel_size=3, stride=2, expand_ratio=6),
                MBConvBlock(80, 80, kernel_size=3, stride=1, expand_ratio=6),
                MBConvBlock(80, 112, kernel_size=3, stride=1, expand_ratio=6),
                MBConvBlock(112, 112, kernel_size=3, stride=1, expand_ratio=6),
                MBConvBlock(112, 192, kernel_size=5, stride=2, expand_ratio=6),
                MBConvBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6),
                MBConvBlock(192, 192, kernel_size=5, stride=1, expand_ratio=6),
                MBConvBlock(192, 320, kernel_size=3, stride=1, expand_ratio=6),
            ]
        )
        self.conv_head = nn.Conv2d(320, 1280, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn0(x)
        x = self.relu(x)
        for block in self.blocks:
            x = block(x)
        x = self.conv_head(x)
        x = self.bn1(x)
        x = self.relu(x)
        return x


# Define Decoder Block
class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Define UNet Decoder
class UnetDecoder(nn.Module):
    def __init__(self):
        super(UnetDecoder, self).__init__()
        self.center = nn.Identity()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(432, 256),
                DecoderBlock(296, 128),
                DecoderBlock(152, 64),
                DecoderBlock(96, 32),
                DecoderBlock(32, 16),
            ]
        )

    def forward(self, x):
        x = self.center(x)
        for block in self.blocks:
            x = block(x)
        return x


# Define Segmentation Head
class SegmentationHead(nn.Module):
    def __init__(self):
        super(SegmentationHead, self).__init__()
        self.conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


# Define UNet
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = UnetDecoder()
        self.segmentation_head = SegmentationHead()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.segmentation_head(x)
        return x
