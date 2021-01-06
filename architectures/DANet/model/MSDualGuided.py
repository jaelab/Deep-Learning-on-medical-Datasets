from .PreAttentionModule import *
from .ResNextModule import *
from .SemGuidedAttModule import *


class MSDualGuided(nn.Module):
    def __init__(self):
        super(MSDualGuided, self).__init__()

        self.resnext_module = ResNextModule()
        self.preattention_module = PreAttentionModule()
        self.semguidedatt_module = SemGuidedAttModule()
        self.apply(weights_init)

    def forward(self, x):
        f1, f2, f3, f4 = self.resnext_module(x)
        fsp, fms, fsms, predict1 = self.preattention_module(f1, f2, f3, f4)
        semVector1, semVector2, fai, semModule1, semModule2, predict2 = self.semguidedatt_module(fsp, fms, fsms)

        predict1_1 = F.upsample(predict1[0], size=x.size()[2:], mode='bilinear')
        predict1_2 = F.upsample(predict1[1], size=x.size()[2:], mode='bilinear')
        predict1_3 = F.upsample(predict1[2], size=x.size()[2:], mode='bilinear')
        predict1_4 = F.upsample(predict1[3], size=x.size()[2:], mode='bilinear')
        predict1 = (predict1_1, predict1_2, predict1_3, predict1_4)

        predict2_1 = F.upsample(predict2[0], size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2[1], size=x.size()[2:], mode='bilinear')
        predict2_3 = F.upsample(predict2[2], size=x.size()[2:], mode='bilinear')
        predict2_4 = F.upsample(predict2[3], size=x.size()[2:], mode='bilinear')
        predict2 = (predict2_1, predict2_2, predict2_3, predict2_4)

        if self.training:
            return semVector1, semVector2, fsms, fai, semModule1, semModule2, predict1, predict2
        else:
            return (predict2[0] + predict2[1] + predict2[2] + predict2[3]) / 4


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal_(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)