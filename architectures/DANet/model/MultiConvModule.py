from torch import nn
from .InterpolationModule import *


class MultiConvModule(nn.Module):
    def __init__(self,  in_channels, out_channels, attn=True):
        super(MultiConvModule, self).__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Softmax2d() if attn else nn.PReLU()
        )

    def forward(self, x):
        return self.fuse(x)
