import torch.nn as nn
from math import sqrt
import torch
import torch.nn.functional as F
from architectures.SegAN.model.GlobalConvolution import GlobalConvolution
from architectures.SegAN.model.ResidualEncoderBlock import ResidualEncoderBlock
from architectures.SegAN.model.ResidualDecoderBlock import ResidualDecoderBlock

channels = 3
dim = 64


class Segmentor(nn.Module):
    def __init__(self):
        super(Segmentor, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, dim, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv1_res = ResidualEncoderBlock(dim)

        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv2_res = ResidualEncoderBlock(dim * 2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(dim * 2, dim * 4, 5, 2, 2, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv3_res = ResidualEncoderBlock(dim * 4)

        self.conv4 = nn.Sequential(
            nn.Conv2d(dim * 4, dim * 8, 5, 2, 2, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv4_res = ResidualEncoderBlock(dim * 8)

        self.conv5 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(dim * 16),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv5_res = ResidualEncoderBlock(dim * 16)

        self.conv6 = nn.Sequential(
            nn.Conv2d(dim * 16, dim * 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim * 32),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv1 = nn.Sequential(
            nn.Conv2d(dim * 32, dim * 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(dim * 16),
            nn.ReLU(True),
        )
        self.deconv1_res = ResidualDecoderBlock(dim * 16)

        self.deconv2 = nn.Sequential(
            GlobalConvolution(dim * 32, dim * 8, (7, 7)),
            nn.BatchNorm2d(dim * 8),
            nn.ReLU(True),
        )
        self.deconv2_res = ResidualDecoderBlock(dim * 8)

        self.deconv3 = nn.Sequential(
            GlobalConvolution(dim * 16, dim * 4, (7, 7)),
            nn.BatchNorm2d(dim * 4),
            nn.ReLU(True),
        )
        self.deconv3_res = ResidualDecoderBlock(dim * 4)

        self.deconv4 = nn.Sequential(
            GlobalConvolution(dim * 8, dim * 2, (9, 9)),
            nn.BatchNorm2d(dim * 2),
            nn.ReLU(True),
        )
        self.deconv4_res = ResidualDecoderBlock(dim * 2)

        self.deconv5 = nn.Sequential(
            GlobalConvolution(dim * 4, dim, (9, 9)),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )
        self.deconv5_res = ResidualDecoderBlock(dim)

        self.deconv6 = nn.Sequential(
            GlobalConvolution(dim * 2, dim, (11, 11)),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
        )
        self.deconv6_res = ResidualDecoderBlock(dim)
        self.deconv7 = nn.Sequential(
            nn.Conv2d(dim, 1, 5, 1, 2, bias=False),
        )

    def forward(self, input):
        enc1 = self.conv1(input)
        enc1 = self.conv1_res(enc1)
        enc2 = self.conv2(enc1)
        enc2 = self.conv2_res(enc2)
        enc3 = self.conv3(enc2)
        enc3 = self.conv3_res(enc3)
        enc4 = self.conv4(enc3)
        enc4 = self.conv4_res(enc4) + enc4
        enc5 = self.conv5(enc4)
        enc5 = self.conv5_res(enc5)
        enc6 = self.conv6(enc5)

        dec1 = self.deconv1(enc6)
        dec1 = self.deconv1_res(dec1)
        dec1 = torch.cat([enc5, dec1], 1)
        dec1 = F.interpolate(dec1, size=enc4.size()[2:], mode='bilinear')
        dec2 = self.deconv2(dec1)
        dec2 = self.deconv2_res(dec2) + dec2
        dec2 = torch.cat([enc4, dec2], 1)
        dec2 = F.interpolate(dec2, size=enc3.size()[2:], mode='bilinear')
        dec3 = self.deconv3(dec2)
        dec3 = self.deconv3_res(dec3)
        dec3 = torch.cat([enc3, dec3], 1)
        dec3 = F.interpolate(dec3, size=enc2.size()[2:], mode='bilinear')
        dec4 = self.deconv4(dec3)
        dec4 = self.deconv4_res(dec4)
        dec4 = torch.cat([enc2, dec4], 1)
        dec4 = F.interpolate(dec4, size=enc1.size()[2:], mode='bilinear')
        dec5 = self.deconv5(dec4)
        dec5 = self.deconv5_res(dec5)
        dec5 = torch.cat([enc1, dec5], 1)
        dec5 = F.interpolate(dec5, size=input.size()[2:], mode='bilinear')
        dec6 = self.deconv6(dec5)
        dec6 = self.deconv6_res(dec6)
        dec7 = self.deconv7(dec6)

        return dec7
