import torch
from torch import nn


class ResNetBlock(nn.Module):
    def __init__(self, in_channel):
        super(ResNetBlock, self).__init__()

        self.skip_connection = nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3, 3), stride=1, padding=1)

        self.direct_connection = nn.Sequential(
            nn.GroupNorm(8, in_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.GroupNorm(8, in_channel),
            nn.ReLU(),
            nn.Conv3d(in_channels=in_channel, out_channels=in_channel, kernel_size=(3, 3, 3), stride=1, padding=1)
        )

    def forward(self, x):
        skip_connection_out = self.skip_connection(x)
        direct_out = self.direct_connection(x)

        out = skip_connection_out + direct_out

        return out
