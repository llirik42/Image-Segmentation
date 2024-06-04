import torch
from torch import nn

from .config import config

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(kernel_size=3,
                               in_channels=in_channels,
                               out_channels=out_channels,
                               padding=1,
                               device=dev
                               )
        self.bn1 = nn.BatchNorm2d(out_channels, device=dev)

        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(kernel_size=3,
                               in_channels=out_channels,
                               out_channels=out_channels,
                               padding=1,
                               device=dev
                               )
        self.bn2 = nn.BatchNorm2d(out_channels, device=dev)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class Encoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.encoder_blocks = nn.ModuleList(
            [
                Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        intermediate_block_outputs = []

        for block in self.encoder_blocks:
            x = block(x)
            intermediate_block_outputs.append(x)
            x = self.pool(x)

        return x, intermediate_block_outputs


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.channels = channels

        self.up_conv = nn.ModuleList(
            [
                nn.ConvTranspose2d(channels[i], channels[i + 1], 2, 2, device=dev)
                for i in range(len(channels) - 1)
            ]
        )

        self.bn = nn.ModuleList([
            nn.BatchNorm2d(channels[i + 1], device=dev)
            for i in range(len(channels) - 1)
        ])

        self.blocks = nn.ModuleList(
            [
                Block(channels[i], channels[i + 1])
                for i in range(len(channels) - 1)
            ]
        )

    def forward(self, x, encoder_features):
        for i in range(len(self.channels) - 1):
            x = self.up_conv[i](x)
            x = self.bn[i](x)

            x = torch.cat([x, encoder_features[config['height'] - i - 1]], dim=1)
            x = self.blocks[i](x)

        return x


class UNet(nn.Module):
    """
    UNet for binary segmentation
    """

    def __init__(self,
                 encoder_channels=(3, 32, 64, 128, 256),
                 decoder_channels=(512, 256, 128, 64, 32)
                 ):
        super().__init__()

        self.encoder = Encoder(encoder_channels)

        self.bottle_conv1 = nn.Conv2d(kernel_size=3,
                                      in_channels=encoder_channels[-1],
                                      out_channels=decoder_channels[0],
                                      padding=1,
                                      device=dev
                                      )
        self.bn = nn.BatchNorm2d(decoder_channels[0], device=dev)
        self.relu = nn.ReLU()
        self.bottle_conv2 = nn.Conv2d(kernel_size=3,
                                      in_channels=decoder_channels[0],
                                      out_channels=decoder_channels[0],
                                      padding=1,
                                      device=dev
                                      )

        self.decoder = Decoder(decoder_channels)

        self.conv_out = nn.Conv2d(kernel_size=1,
                                  in_channels=decoder_channels[-1],
                                  out_channels=1,
                                  device=dev)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # encoder
        x, encoder_features = self.encoder(x)

        # bottleneck
        x = self.relu(self.bn(self.bottle_conv1(x)))
        x = self.relu(self.bn(self.bottle_conv2(x)))

        # decoder
        x = self.decoder(x, encoder_features)

        #
        x = self.conv_out(x)
        x = self.sigmoid(x)

        return x
