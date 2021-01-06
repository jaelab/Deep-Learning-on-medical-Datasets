from keras.layers import *

class UpSamplingBlock():
    def __init__(self, out_channels):
        self.out_channels = out_channels
        self.convTransp1 = Conv2DTranspose(self.out_channels,
                                           kernel_size=2,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer='he_normal')
        self.batchNorm = BatchNormalization(axis=3)
        self.ReLUActivation = Activation('relu')

    def __call__(self, input_values):
        convTransp1 = self.convTransp1(input_values)
        batchNorm = self.batchNorm(convTransp1)
        ReLUActivation = self.ReLUActivation(batchNorm)
        return ReLUActivation

