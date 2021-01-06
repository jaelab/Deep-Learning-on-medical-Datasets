import torch.nn as nn
from math import sqrt
import torch
from architectures.SegAN.model.GlobalConvolution import GlobalConvolution

channel_dim = 3
dim = 64


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.conv1 = nn.Sequential(
            GlobalConvolution(channel_dim, dim, (13, 13), 2),
            nn.BatchNorm2d(dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            GlobalConvolution(dim, dim * 2, (11, 11), 2),
            nn.BatchNorm2d(dim * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            GlobalConvolution(dim * 2, dim * 4, (9, 9), 1),
            nn.BatchNorm2d(dim * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            GlobalConvolution(dim * 4, dim * 8, (7, 7), 1),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 8, 4, 1, 2, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv6 = nn.Sequential(
            nn.Conv2d(dim * 8, dim * 8, 3, 2, 2, bias=False),
            nn.BatchNorm2d(dim * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        batch_size = input.size()[0]
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        # Concatenate output
        output = torch.cat((input.view(batch_size, -1), 1 * out1.view(batch_size, -1),
                            2 * out2.view(batch_size, -1), 2 * out3.view(batch_size, -1),
                            2 * out4.view(batch_size, -1), 2 * out5.view(batch_size, -1),
                            4 * out6.view(batch_size, -1)), 1)
        return output
