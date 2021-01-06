from architectures.NVDLMED.model.ResNetBlock import *


class DecoderGT(nn.Module):
    def __init__(self, in_channels=256, output_channels=3):
        super(DecoderGT, self).__init__()

        self.out_up_1_channels = in_channels // 2
        self.out_up_2_channels = self.out_up_1_channels // 2
        self.out_up_3_channels = self.out_up_2_channels // 2

        self.first_upsample3d = UpsampleBlock(in_channels, self.out_up_1_channels)
        self.first_resnetblock = ResNetBlock(in_channel=self.out_up_1_channels)

        self.second_upsample3d = UpsampleBlock(self.out_up_1_channels, self.out_up_2_channels)
        self.second_resnetblock = ResNetBlock(in_channel=self.out_up_2_channels)

        self.third_upsample3d = UpsampleBlock(self.out_up_2_channels, self.out_up_3_channels)
        self.third_resnetblock = ResNetBlock(in_channel=self.out_up_3_channels)

        self.output_gt = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=output_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3, x4):
        # First Decoder ResNetBlock (output filters = 128)
        x = self.first_upsample3d(x4)
        x = torch.add(x, x3)
        x = self.first_resnetblock(x)

        # Second Decoder ResNetBlock (output filters = 64)
        x = self.second_upsample3d(x)
        x = torch.add(x, x2)
        x = self.second_resnetblock(x)

        # Third Decoder ResNetBlock (output filters = 32)
        x = self.third_upsample3d(x)
        x = torch.add(x, x1)
        x = self.third_resnetblock(x)

        # Output Block
        out_gt = self.output_gt(x)

        return out_gt


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.up_sample = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1, 1), stride=1),
            nn.Upsample(scale_factor=2, mode='trilinear'))

    def forward(self, x):
        return self.up_sample(x)
