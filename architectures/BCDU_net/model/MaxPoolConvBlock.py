from architectures.BCDU_net.model.PairBlock import *


class MaxPoolConvBlock():
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.maxPool1 = MaxPooling2D(pool_size=(2, 2))
        self.pairBlock1 = PairBlock(self.out_channels)


    def __call__(self, input_values):
        maxPool1 = self.maxPool1(input_values)
        pairBlock1 = self.pairBlock1(maxPool1)
        return pairBlock1
