from torch import nn
import torch
from .SementicModule import *
from .PosChanAttModule import *


class SemGuidedAttModule(nn.Module):
    def __init__(self):
        super(SemGuidedAttModule, self).__init__()

        self.semModule1 = SementicModule(128)
        self.sem_att_block1_1 = SemAttBlock(64, self.semModule1)
        self.sem_att_block1_2 = SemAttBlock(64, self.semModule1)
        self.sem_att_block1_3 = SemAttBlock(64, self.semModule1)
        self.sem_att_block1_4 = SemAttBlock(64, self.semModule1)

        self.semModule2 = SementicModule(128)
        self.sem_att_block2_1 = SemAttBlock(64, self.semModule2)
        self.sem_att_block2_2 = SemAttBlock(64, self.semModule2)
        self.sem_att_block2_3 = SemAttBlock(64, self.semModule2)
        self.sem_att_block2_4 = SemAttBlock(64, self.semModule2)

        self.predict1 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 5, kernel_size=1)
        self.predict4 = nn.Conv2d(64, 5, kernel_size=1)

    def forward(self, fsp, fms, fsms):
        att1_1, semVector1_1, semModule1_1 = self.sem_att_block1_1(fsms[0])
        att1_2, semVector1_2, semModule1_2 = self.sem_att_block1_2(fsms[1])
        att1_3, semVector1_3, semModule1_3 = self.sem_att_block1_3(fsms[2])
        att1_4, semVector1_4, semModule1_4 = self.sem_att_block1_4(fsms[3])

        fa1 = torch.cat((fsp[0], att1_1 * fms), 1)
        att2_1, semVector2_1, semModule2_1 = self.sem_att_block2_1(fa1)

        fa2 = torch.cat((fsp[1], att1_2 * fms), 1)
        att2_2, semVector2_2, semModule2_2 = self.sem_att_block2_2(fa2)

        fa3 = torch.cat((fsp[2], att1_3 * fms), 1)
        att2_3, semVector2_3, semModule2_3 = self.sem_att_block2_3(fa3)

        fa4 = torch.cat((fsp[3], att1_4 * fms), 1)
        att2_4, semVector2_4, semModule2_4 = self.sem_att_block2_4(fa4)

        predict1 = self.predict1(att2_1)
        predict2 = self.predict2(att2_2)
        predict3 = self.predict3(att2_3)
        predict4 = self.predict4(att2_4)

        return (semVector1_1, semVector1_2, semVector1_3, semVector1_4), \
               (semVector2_1, semVector2_2, semVector2_3, semVector2_4), \
               (fa1, fa2, fa3, fa4), \
               (semModule1_1, semModule1_2, semModule1_3, semModule1_4), \
               (semModule2_1, semModule2_2, semModule2_3, semModule2_4), \
               (predict1, predict2, predict3, predict4)


class SemAttBlock(nn.Module):
    def __init__(self, in_channels, sementic_module):
        super(SemAttBlock, self).__init__()

        self.att = PosChanAttModule(in_channels)
        self.semModule = sementic_module

        self.conv_sem = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_dot = nn.Conv2d(64, 64, 1)

    def forward(self, x):
        semVector, semModule = self.semModule(x)
        semModule_conv = self.conv_sem(semModule)

        cam_pam_att = self.att(x)
        attention = self.conv_dot(cam_pam_att * semModule_conv)

        return attention, semVector, semModule
