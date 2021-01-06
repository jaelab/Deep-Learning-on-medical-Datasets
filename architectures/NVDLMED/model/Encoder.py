from architectures.NVDLMED.model.ResNetBlock import *


class Encoder(nn.Module):
    def __init__(self, in_channels, start_channels=32):
        super(Encoder, self).__init__()

        self.start_channels = start_channels
        self.down_channels_1 = 2 * self.start_channels
        self.down_channels_2 = 2 * self.down_channels_1
        self.down_channels_3 = 2 * self.down_channels_2

        self.blue_block = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=self.start_channels, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.Dropout3d(p=0.2))

        self.first_encoder = ResNetBlock(in_channel=self.start_channels)

        self.second_encoder = nn.Sequential(
            nn.Conv3d(in_channels=self.start_channels, out_channels=self.down_channels_1, kernel_size=(3, 3, 3), stride=2, padding=1),
            ResNetBlock(in_channel=self.down_channels_1),
            ResNetBlock(in_channel=self.down_channels_1))

        self.third_encoder = nn.Sequential(
            nn.Conv3d(in_channels=self.down_channels_1, out_channels=self.down_channels_2, kernel_size=(3, 3, 3), stride=2, padding=1),
            ResNetBlock(in_channel=self.down_channels_2),
            ResNetBlock(in_channel=self.down_channels_2))

        self.fourth_encoder = nn.Sequential(
            nn.Conv3d(in_channels=self.down_channels_2, out_channels=self.down_channels_3, kernel_size=(3, 3, 3), stride=2, padding=1),
            ResNetBlock(in_channel=self.down_channels_3),
            ResNetBlock(in_channel=self.down_channels_3),
            ResNetBlock(in_channel=self.down_channels_3),
            ResNetBlock(in_channel=self.down_channels_3))

    def forward(self, x):
        # Initial Block
        x = self.blue_block(x)

        # First Encoder ResNetBlock (output filters = 32)
        x1 = self.first_encoder(x)

        # Second Encoder ResNetBlock (output filters = 64)
        x2 = self.second_encoder(x1)

        # Third Encoder ResNetBlock (output filters = 128)
        x3 = self.third_encoder(x2)

        # Fourth Encoder ResNetBlock (output filters = 256)
        x4 = self.fourth_encoder(x3)

        return x1, x2, x3, x4
