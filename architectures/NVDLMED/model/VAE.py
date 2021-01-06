from torch import nn
from architectures.NVDLMED.model.ResNetBlock import *
from architectures.NVDLMED.model.DecoderGT import UpsampleBlock
from torch.distributions.normal import Normal


class VAE(nn.Module):
    def __init__(self, input_shape=(256, 20, 24, 16), output_channels=4, vd_out_channels=16, vae_vector_dim=256):
        super(VAE, self).__init__()

        # Linear reduction dimensions
        self.linear_reduc_dim = int(vd_out_channels * (input_shape[1] // 2) * (input_shape[2] // 2) * (input_shape[3] // 2))
        self.start_channels = input_shape[0]
        self.up_channels_1 = self.start_channels // 2
        self.up_channels_2 = self.up_channels_1 // 2
        self.up_channels_3 = self.up_channels_2 // 2

        self.vd_block = nn.Sequential(
            nn.GroupNorm(8, 256),
            nn.ReLU(),
            nn.Conv3d(in_channels=self.start_channels, out_channels=vd_out_channels, kernel_size=(3, 3, 3), stride=2, padding=1),
            Flatten(),
            nn.Linear(self.linear_reduc_dim, vae_vector_dim))

        self.vddraw_block = SamplingBlock(input_shape=vae_vector_dim)

        self.vu_block = nn.Sequential(
            nn.Linear(128, self.linear_reduc_dim),
            nn.ReLU(),
            View((-1, vd_out_channels, input_shape[1] // 2, input_shape[2] // 2, input_shape[3] // 2)),
            UpsampleBlock(in_channels=vd_out_channels, out_channels=self.start_channels))

        self.vup_resnet_block2 = nn.Sequential(
            UpsampleBlock(in_channels=self.start_channels, out_channels=self.up_channels_1),
            ResNetBlock(in_channel=self.up_channels_1))

        self.vup_resnet_block1 = nn.Sequential(
            UpsampleBlock(in_channels=self.up_channels_1, out_channels=self.up_channels_2),
            ResNetBlock(in_channel=self.up_channels_2))

        self.vup_resnet_block0 = nn.Sequential(
            UpsampleBlock(in_channels=self.up_channels_2, out_channels=self.up_channels_3),
            ResNetBlock(in_channel=self.up_channels_3))

        self.output_vae = nn.Conv3d(in_channels=self.up_channels_3, out_channels=output_channels, kernel_size=(1, 1, 1), stride=1)

    def forward(self, x):
        # VD Block (Reducing dimensionality of the data)
        x = self.vd_block(x)

        # Sampling Block
        x, mu, logvar = self.vddraw_block(x)

        # VU BLock (Upsizing back to a depth of 256)
        x = self.vu_block(x)

        # First Decoder ResNetBlock (output filters = 128)
        x = self.vup_resnet_block2(x)

        # Second Decoder ResNetBlock (output filters = 64)
        x = self.vup_resnet_block1(x)

        # Third Decoder ResNetBlock (output filters = 32)
        x = self.vup_resnet_block0(x)

        # Output Block
        out_vae = self.output_vae(x)

        return out_vae, mu, logvar


class Flatten(nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, -1)

        return out


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class SamplingBlock(nn.Module):
    def __init__(self, input_shape=256):
        super(SamplingBlock, self).__init__()

        self.input_shape = input_shape

    @staticmethod
    def sampling(z_mean, z_var):
        epsilon = torch.rand_like(z_mean).cuda()

        return z_mean + torch.exp(0.5 * z_var) * epsilon

    def forward(self, x):
        z_mean = x[:, :(self.input_shape // 2)]
        z_var = x[:, (self.input_shape // 2):]

        out = self.sampling(z_mean, z_var)

        return out, z_mean, z_var
