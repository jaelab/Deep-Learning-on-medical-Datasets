from torch import nn
from architectures.NVDLMED.model.Encoder import *
from architectures.NVDLMED.model.DecoderGT import *
from architectures.NVDLMED.model.VAE import *


class NVDLMED(nn.Module):
    def __init__(self, input_shape=(4, 160, 192, 128), output_gt=3, output_vae=4):
        super(NVDLMED, self).__init__()

        self.vae_in_resolution = (256, input_shape[1] // 8, input_shape[2] // 8, input_shape[3] // 8)

        self.encoder = Encoder(in_channels=input_shape[0])
        self.decoder_gt = DecoderGT(output_channels=output_gt)
        self.vae = VAE(input_shape=self.vae_in_resolution, output_channels=output_vae)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        decoded_gt = self.decoder_gt(x1, x2, x3, x4)

        if self.training:
            decoded_vae, mu, logvar = self.vae(x4)
            return decoded_gt, decoded_vae, mu, logvar

        else:
            return decoded_gt
