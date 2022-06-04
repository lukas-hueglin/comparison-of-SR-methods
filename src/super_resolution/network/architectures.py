### architectures.py ###
# This module includes the build functions, which return a specific tensorflow
# model architecture. Every function with the make_ prefix will return such a network.
##

import tensorflow as tf

from keras import layers, Sequential
from keras import Model

# A placeholder function for the SRResNet architecture
def make_SRResNet(input_res):
    def residual_block(x, i):
        # skip connection
        skip = x

        # layers
        x = layers.Conv2D(64, (3, 3), strides=1, padding='same', name='ResB_'+ f"{i+1:02d}" +'_Conv_01')(x)
        x = layers.BatchNormalization(name='ResB_'+ f"{i+1:02d}" +'_BN_01')(x)

        x = layers.PReLU(name='ResB_'+ f"{i+1:02d}" +'_PReLU')(x)

        x = layers.Conv2D(64, (3, 3), strides=1, padding='same', name='ResB_'+ f"{i+1:02d}" +'_Conv_02')(x)
        x = layers.BatchNormalization(name='ResB_'+ f"{i+1:02d}" +'_BN_02')(x)

        # add skip connection
        x = layers.Add(name='ResB_'+ f"{i+1:02d}" +'_Sum')([x, skip])

        return x

    def upscale_block(x, i):
        x = layers.Conv2D(256, (3, 3), strides=1, padding='same', name='UpB_'+ f"{i+1:02d}" +'_Conv')(x)
        x = layers.Lambda(lambda x: tf.nn.depth_to_space(x, 2), name='UpB_'+ f"{i+1:02d}" +'_PixShuf')(x) # pixelshuffle
        x = layers.PReLU(name='UpB_'+ f"{i+1:02d}" +'_PReLU')(x)
    
        return x


    # build network
    input = layers.Input(shape=(input_res, input_res, 3), name='LR_Input')

    # first pre Residual layers
    x = layers.Conv2D(64, (9, 9), strides=1, padding='same', name='Conv_01')(input)
    x = layers.PReLU(name='PReLU_01')(x)

    # skip connection over all residual blocks
    skip = x

    # b residual blocks
    b = 16

    for i in range(b):
        x = residual_block(x, i)

    # layers after residual blocks
    x = layers.Conv2D(64, (3, 3), strides=1, padding='same', name='Conv_02')(x)
    x = layers.BatchNormalization(name='BN_01')(x)
    x = layers.Add(name='Sum_01')([x, skip])

    # two upscaling blocks
    for i in range(2):
        x = upscale_block(x, i)

    # ending conv
    y = layers.Conv2D(3, (9, 9), strides=1, padding='same', name='Conv_03')(x)

    return Model(inputs=input, outputs=y, name='SRResNet')
    

# A placeholder function for the SRGan discriminator
def make_SRGAN_disc(input_res):
    def block(x, filters, stride, i):
        x = layers.Conv2D(filters, (3, 3), strides=stride, name='B_'+ f"{i:02d}" +'_Conv')(x)
        x = layers.BatchNormalization(name='B_'+ f"{i:02d}" +'_BN')(x)
        x = layers.LeakyReLU(alpha=0.2, name='B_'+ f"{i:02d}" +'_LeakyReLU')(x)

        return x

    # build network
    input = layers.Input(shape=(input_res, input_res, 3), name='Gen_Input')

    # first layers
    x = layers.Conv2D(64, (3, 3), strides=1, name='Conv')(input)
    x = layers.LeakyReLU(alpha=0.2,name='LeakyReLU_01')(x)

    # 7 blocks
    x = block(x, 64, 2, 1)
    x = block(x, 128, 1, 2)
    x = block(x, 128, 2, 3)
    x = block(x, 256, 1, 4)
    x = block(x, 256, 2, 5)
    x = block(x, 512, 1, 6)
    x = block(x, 512, 2, 7)

    # Dense layers
    x = layers.Dense(1024, name='Dense_01')(x)
    x = layers.LeakyReLU(alpha=0.2,name='LeakyReLU_02')(x)
    y = layers.Dense(1, name='Dense_02', activation='sigmoid')(x)

    return Model(inputs=input, outputs=y, name='SRGan_disc')


# This network is just for test purposes and isn't build for achieving great accuracies
def make_Demo(input_res):
    network = Sequential(name='SRDemo')

    network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_res, input_res, 3)))
    network.add(layers.Conv2D(64, (5, 5), activation='relu'))
    network.add(layers.Conv2D(64, (3, 3), activation='relu'))

    network.add(layers.Conv2DTranspose(32, (3, 3), activation='relu'))
    network.add(layers.Conv2DTranspose(64, (5, 5), activation='relu'))
    network.add(layers.Conv2DTranspose(3, (3, 3), activation='relu'))

    return network