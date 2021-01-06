from keras.layers import *

class PairBlock():
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.conv1 = Conv2D(self.out_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')
        self.conv2 = Conv2D(self.out_channels, 3, activation='relu', padding='same', kernel_initializer='he_normal')

    def __call__(self, input_values):
        conv1 = self.conv1(input_values)
        conv2 = self.conv2(conv1)
        return conv2
