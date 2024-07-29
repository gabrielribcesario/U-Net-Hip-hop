import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, MaxPooling2D,  UpSampling2D, BatchNormalization, Activation, Concatenate)

class UNet_SegBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kwargs):
        super().__init__()
        activation = kwargs.copy().pop('activation', None)
        if activation is None: activation = 'relu'
        self.conv1 = Conv2D(filters, (3,3), **kwargs)
        self.bn1 = BatchNormalization()
        self.act1 = Activation(activation)
        self.conv2 = Conv2D(filters, (3,3), **kwargs)
        self.bn2 = BatchNormalization()
        self.act2 = Activation(activation)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act2(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config 

class UNet_UpConv(tf.keras.layers.Layer):
    def __init__(self, filters, kwargs):
        super().__init__()
        self.up1 = UpSampling2D((2,2))
        self.conv1 = Conv2D(filters, (2,2), **kwargs)

    def call(self, x):
        x = self.up1(x)
        x = self.conv1(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config 

class UNet(tf.keras.layers.Layer):
    def __init__(self, output_maps, output_act='softmax', activation='relu', 
                 kernel_initializer='HeUniform', bias_initializer='zeros', 
                 kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                 **kwargs):
        param = locals(); param['padding'] = 'same'; remove = ('__class__', 'output_maps', 'self', 'kwargs', 'output_act')
        for i in remove: param.pop(i)
        del remove
        super().__init__()
        self.output_maps = output_maps
        ### Encoder ###
        # 1st step
        self.ds1 = UNet_SegBlock(64, param)
        self.identity1 = Activation('linear') # (reusing the same Activation/using an = sign) might (overwrite past gradients/create a reference of the original tensor), which is why I'm creating an (attribute/tensor) for each (identity/residual)
        self.mp1 = MaxPooling2D((2,2))
        # 2nd step
        self.ds2 = UNet_SegBlock(128, param)
        self.identity2 = Activation('linear')
        self.mp2 = MaxPooling2D((2,2))
        # 3rd step
        self.ds3 = UNet_SegBlock(256, param)
        self.identity3 = Activation('linear')
        self.mp3 = MaxPooling2D((2,2))
        # 4th step
        self.ds4 = UNet_SegBlock(512, param)
        self.identity4 = Activation('linear')
        self.mp4 = MaxPooling2D((2,2))
        ### Bottleneck ###
        self.btn = UNet_SegBlock(1024, param)
        ### Decoder ###
        # 1st step
        self.upconv1 = UNet_UpConv(512, param)
        self.conc1 = Concatenate(axis=-1)
        self.us1 = UNet_SegBlock(512, param)
        # 2nd step
        self.upconv2 = UNet_UpConv(256, param)
        self.conc2 = Concatenate(axis=-1)
        self.us2 = UNet_SegBlock(256, param)
        # 3rd step
        self.upconv3 = UNet_UpConv(128, param)
        self.conc3 = Concatenate(axis=-1)
        self.us3 = UNet_SegBlock(128, param)
        # 4th step
        self.upconv4 = UNet_UpConv(64, param)
        self.conc4 = Concatenate(axis=-1)
        self.us4 = UNet_SegBlock(64, param)
        ## Thresholding ###
        param['kernel_initializer'] = 'GlorotUniform'; 
        if self.output_maps == 1: param['activation'] = 'sigmoid'
        else: param['activation'] = output_act
        self.thr = Conv2D(self.output_maps, (1,1), dtype=tf.float32, **param)

    def call(self, x):
        ### Encoder ###
        # 1st step
        x = self.ds1(x)
        res1 = self.identity1(x)
        x = self.mp1(x)
        # 2nd step
        x = self.ds2(x)
        res2 = self.identity2(x)
        x = self.mp2(x)
        # 3rd step
        x = self.ds3(x)
        res3 = self.identity3(x)
        x = self.mp3(x)
        # 4th step
        x = self.ds4(x)
        res4 = self.identity4(x)
        x = self.mp4(x)
        ### Bottleneck ###
        x = self.btn(x)
        ### Decoder ###
        # 1st step
        x = self.upconv1(x)
        x = self.conc1([res4, x])
        x = self.us1(x)
        # 2nd step
        x = self.upconv2(x)
        x = self.conc2([res3, x])
        x = self.us2(x)
        # 3rd step
        x = self.upconv3(x)
        x = self.conc3([res2, x])
        x = self.us3(x)
        # 4th step
        x = self.upconv4(x)
        x = self.conc4([res1, x])
        x = self.us4(x)
        ### Thresholding ###
        x = self.thr(x)
        return x

    def get_config(self):
        config = super().get_config()
        return config 
