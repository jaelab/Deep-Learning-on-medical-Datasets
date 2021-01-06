import torch
from .MultiConvModule import *


class PreAttentionModule(nn.Module):
    def __init__(self):
        super(PreAttentionModule, self).__init__()

        self.interpolation = InterpolationModule()
        self.predict4 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 5, kernel_size=1)
        self.fuse1 = MultiConvModule(256, 64, False)

    def forward(self, f1, f2, f3, f4):
        f1p, f2p, f3p, f4p = self.interpolation(f1, f2, f3, f4)

        predict4 = self.predict4(f4p)
        predict3 = self.predict3(f3p)
        predict2 = self.predict2(f2p)
        predict1 = self.predict1(f1p)

        fms = self.fuse1(torch.cat((f1p, f2p, f3p, f4p), 1)) # J'ai mis l'ordre inverse (comme dans le paper)

        f1ms = torch.cat((f1p, fms), 1)
        f2ms = torch.cat((f2p, fms), 1)
        f3ms = torch.cat((f3p, fms), 1)
        f4ms = torch.cat((f4p, fms), 1)

        return (f1p, f2p, f3p, f4p), fms, (f1ms, f2ms, f3ms, f4ms), (predict1, predict2, predict3, predict4)
