from torch import nn
from torchvision.models import resnext101_32x8d
from .ResNextModule import *
from torch.nn import functional as F


class InterpolationModule(nn.Module):
    def __init__(self):
        super(InterpolationModule, self).__init__()

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.PReLU()
        )

    def forward(self, f1, f2, f3, f4):
        f4p = F.upsample(self.down4(f4), size=f1.size()[2:], mode='bilinear')
        f3p = F.upsample(self.down3(f3), size=f1.size()[2:], mode='bilinear')
        f2p = F.upsample(self.down2(f2), size=f1.size()[2:], mode='bilinear')
        f1p = self.down1(f1)

        return f1p, f2p, f3p, f4p
