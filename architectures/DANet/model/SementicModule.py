from torch import nn
import torch
from .EncoderBlock import *
from .DecoderBlock import *
from torch.nn import functional as F


class SementicModule(nn.Module):
    def __init__(self, in_channels):
        super(SementicModule, self).__init__()

        self.in_channels = in_channels

        self.enc1 = EncoderBlock(in_channels, in_channels*2)
        self.enc2 = EncoderBlock(in_channels*2, in_channels*4)
        self.dec2 = DecoderBlock(in_channels*4, in_channels*2, in_channels*2)
        self.dec1 = DecoderBlock(in_channels*2, in_channels, in_channels)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2(enc2)
        dec1 = self.dec1(F.upsample(dec2, enc1.size()[2:], mode='bilinear'))

        return enc2.view(-1), dec1
