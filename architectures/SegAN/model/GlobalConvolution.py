import torch.nn as nn
from math import sqrt


class GlobalConvolution(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1):
        super(GlobalConvolution, self).__init__()
        kernel_1 = kernel_size[0]
        kernel_2 = kernel_size[1]

        padding_1 = (kernel_1 - 1) // 2
        padding_2 = (kernel_2 - 1) // 2

        self.conv_1 = nn.Conv2d(in_dim, out_dim, kernel_size=(kernel_1, 1), padding=(padding_1, 0), stride=stride)
        self.conv_2 = nn.Conv2d(out_dim, out_dim, kernel_size=(1, kernel_2), padding=(0, padding_2), stride=stride)
        self.conv_3 = nn.Conv2d(in_dim, out_dim, kernel_size=(1, kernel_2), padding=(0, padding_2), stride=stride)
        self.conv_4 = nn.Conv2d(out_dim, out_dim, kernel_size=(kernel_1, 1), padding=(padding_1, 0), stride=stride)

    def forward(self, x):
        x_1 = self.conv_1(x)
        x_1 = self.conv_2(x_1)

        x_2 = self.conv_3(x)
        x_2 = self.conv_4(x_2)

        return x_1 + x_2
