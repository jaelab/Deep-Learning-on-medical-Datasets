from .PairBlock import *

class DenseBlock():
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.pairBlock1 = PairBlock(self.out_channels)
        self.dropout1 = Dropout(0.5)

    def __call__(self, input_values):
        pairBlock1 = self.pairBlock1(input_values)
        dropout1 = self.dropout1(pairBlock1)
        return dropout1