from torch import nn
import torch


class PosChanAttModule(nn.Module):
    def __init__(self, in_channels):
        super(PosChanAttModule, self).__init__()

        self.cam = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            CAM(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

        self.pam = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            PAM(in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU()
        )

    def forward(self, x):
        attn_pam = self.pam(x)
        attn_cam = self.cam(x)

        return attn_pam + attn_cam


class CAM(nn.Module):
    def __init__(self, input_channels):
        super(CAM, self).__init__()

        self.input_channels = input_channels

        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, H, W = x.size()

        proj_query = x.view(batch, C, -1)
        proj_key = x.view(batch, C, -1).permute(0, 2, 1)

        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(batch, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(batch, C, H, W)

        out = self.gamma * out + x

        return out


class PAM(nn.Module):
    def __init__(self, input_channels):
        super(PAM, self).__init__()

        self.input_channels = input_channels

        self.query_conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch, C, H, W = x.size()

        proj_query = self.query_conv(x).view(batch, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(batch, -1, W * H)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(batch, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch, C, H, W)
        out = self.gamma * out + x

        return out

