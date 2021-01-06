from .UpSamplingBlock import *
from .MaxPoolConvBlock import *
from .DenseBlock import *
from keras.optimizers import *
from keras.models import Model

class BCDU_net(Model):
    def __init__(self, input_shape):
        super(BCDU_net, self).__init__()
        #Input block
        self.input_block = PairBlock(64)
        #self.maxPoolBlock1
        self.maxPoolBlock1 = MaxPoolConvBlock(128)

        #self.maxPoolBlock2
        self.maxPoolBlock2 = MaxPoolConvBlock(256)

        #drop3
        self.dropout1 = Dropout(0.5)

        #pool3
        self.maxPoolLayer1 = MaxPooling2D(pool_size=(2, 2))

        #Dense block 1
        self.denseBlock1 = DenseBlock(512)

        #Dense block 2
        self.denseBlock2 = DenseBlock(512)

        #Dense block 3
        self.denseBlock3 = DenseBlock(512)

        #Up-Sampling 1
        self.upSampling1 = UpSamplingBlock(256)

        #Reshape 1
        self.reshape1 = Reshape(target_shape=(1,
                                              np.int32(input_shape[0]/4),
                                              np.int32(input_shape[0]/4),
                                              256))
        #Reshape 2
        self.reshape2 = Reshape(target_shape=(1,
                                              np.int32(input_shape[0]/4),
                                              np.int32(input_shape[0]/4),
                                              256))

        #Convolutional 2D LSTM 1
        self.convLSTM2D1 = ConvLSTM2D(filters=128,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      return_sequences=False,
                                      go_backwards=True,
                                      kernel_initializer='he_normal')

        self.convBlock1 = PairBlock(256)

        #Up-Sampling 2
        self.upSampling2 = UpSamplingBlock(128)

        self.reshapedMaxPoolBlock1 = Reshape(target_shape=(1,
                                                          np.int32(input_shape[0]/2),
                                                          np.int32(input_shape[0]/2),
                                                          128))

        self.reshapedUpSamplingBlock2 = Reshape(target_shape=(1,
                                                          np.int32(input_shape[0]/2),
                                                          np.int32(input_shape[0]/2),
                                                          128))

        #Convolutional 2D LSTM 2
        self.convLSTM2D2 = ConvLSTM2D(filters=64,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      return_sequences=False,
                                      go_backwards=True,
                                      kernel_initializer='he_normal')

        self.convBlock2 = PairBlock(128)

        #Up-Sampling 3
        self.upSampling3 = UpSamplingBlock(64)

        self.reshapedInputBlock = Reshape(target_shape=(1,
                                                   input_shape[0],
                                                   input_shape[0],
                                                   64))

        self.reshapedUpSamplingBlock3 = Reshape(target_shape=(1,
                                                   input_shape[0],
                                                   input_shape[0],
                                                   64))


        #Convolutional 2D LSTM 3
        self.convLSTM2D3 = ConvLSTM2D(filters=32,
                                      kernel_size=(3, 3),
                                      padding='same',
                                      return_sequences=False,
                                      go_backwards=True,
                                      kernel_initializer='he_normal')

        self.convBlock3 = PairBlock(64)
        self.convLayer1 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')
        self.lastLayer = Conv2D(1, 1, activation='sigmoid')

    def __call__(self, input_tensor):
        input_block = self.input_block(input_tensor)
        maxPoolBlock1 = self.maxPoolBlock1(input_block)
        maxPoolBlock2 = self.maxPoolBlock2(maxPoolBlock1)
        # drop3
        dropout1 = self.dropout1(maxPoolBlock2)
        maxPoolLayer1 = self.maxPoolLayer1(maxPoolBlock2)

        # Dense block 1
        denseBlock1 = self.denseBlock1(maxPoolLayer1)

        # Dense block 2
        denseBlock2 = self.denseBlock2(denseBlock1)

        # Dense block 3
        mergedDenseBlocks = concatenate([denseBlock2, denseBlock1], axis=3)
        denseBlock3 = self.denseBlock3(mergedDenseBlocks)

        # Up-Sampling 1
        upSampling1 = self.upSampling1(denseBlock3)

        # Reshape 1
        reshape1 = self.reshape1(dropout1)
        # Reshape 2
        reshape2 = self.reshape2(upSampling1)

        mergedReshapes = concatenate([reshape1, reshape2], axis=1)

        # Convolutional 2D LSTM 1
        convLSTM2D1 = self.convLSTM2D1(mergedReshapes)

        convBlock1 = self.convBlock1(convLSTM2D1)

        # Up-Sampling 2
        upSampling2 = self.upSampling2(convBlock1)

        reshapedMaxPoolBlock1 = self.reshapedMaxPoolBlock1(maxPoolBlock1)

        reshapedUpSamplingBlock2 = self.reshapedUpSamplingBlock2(upSampling2)

        mergedReshapes2 = concatenate([reshapedMaxPoolBlock1, reshapedUpSamplingBlock2], axis=1)

        # Convolutional 2D LSTM 2
        convLSTM2D2 = self.convLSTM2D2(mergedReshapes2)

        convBlock2 = self.convBlock2(convLSTM2D2)

        # Up-Sampling 3
        upSampling3 = self.upSampling3(convBlock2)

        reshapedInputBlock = self.reshapedInputBlock(input_block)

        reshapedUpSamplingBlock3 = self.reshapedUpSamplingBlock3(upSampling3)

        mergedReshapes3 = concatenate([reshapedInputBlock, reshapedUpSamplingBlock3], axis=1)

        # Convolutional 2D LSTM 3
        convLSTM2D3 = self.convLSTM2D3(mergedReshapes3)

        convBlock3 = self.convBlock3(convLSTM2D3)
        convLayer1 = self.convLayer1(convBlock3)
        lastLayer = self.lastLayer(convLayer1)
        return lastLayer
