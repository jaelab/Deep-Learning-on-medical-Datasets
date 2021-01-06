import torch.nn as nn
from math import sqrt


class ResidualDecoderBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualDecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=False)
        self.norm1 = nn.BatchNorm2d(dim * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(dim * 2)
        self.act2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=False)
        self.norm3 = nn.BatchNorm2d(dim)
        self.act3 = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.conv1(x)
        output = self.act1(output)
        output = self.conv2(output)
        output = self.act2(output)
        output = self.conv3(output)
        output = self.act3(output)

        return x + output
