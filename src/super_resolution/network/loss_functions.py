### loss_functions.py ###
# In this module all the loss/cost functions are defined. Every
# function in the project with a _loss suffix is a loss function.
# Every loss function is build with a build_ _loss build function.
##

import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras import Model, layers

from utils import TColors

# This loss function will return the sum of a VGG loss
# function and the gen_loss() function. It is used for the SRGAN preset.
def build_SRGAN_loss(input_res=None):
    # build SRResNet_loss
    SRResNet_loss = build_SRResNet_loss(input_res)
    # build gen_loss
    gen_loss = build_gen_loss()

    def SRGAN_loss(y_true, y_pred, y_disc):
        return SRResNet_loss(y_true, y_pred) + gen_loss(y_disc)

    return SRGAN_loss

# The loss of a VGG loss function (content loss)
# (with help from: https://github.com/deepak112/Keras-SRGAN/blob/master/Utils_model.py)

def build_SRResNet_loss(input_res=None):
    # get shape
    shape = (input_res, input_res, 3)

    # build vgg 19 model
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=shape)

    vgg19.trainable = False
    # set trainable to False
    for l in vgg19.layers:
        l.trainable = False

    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block3_conv4').output)
    model.trainable = False

    # make MSE_loss
    MSE_loss = build_MSE_loss()

    def SRResNet_loss(y_true, y_pred):
        # calculate loss
        return MSE_loss(model(y_true), model(y_pred))

    return SRResNet_loss

# The loss function for the SRGAN generator (adversarial loss)
# (from: https://www.tensorflow.org/tutorials/generative/dcgan)
def build_gen_loss(input_res=None):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def gen_loss(y_disc):
        return cross_entropy(tf.ones_like(y_disc), y_disc)

    return gen_loss

# The loss function of the SRGAN discriminator.
# (from: https://www.tensorflow.org/tutorials/generative/dcgan)
def build_disc_loss(input_res=None):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    def disc_loss(y_real, y_fake):
        real_loss = cross_entropy(tf.ones_like(y_real), y_real)
        fake_loss = cross_entropy(tf.zeros_like(y_fake), y_fake)

        return real_loss + fake_loss

    return disc_loss

# This loss function is packs the default tf MSE into a own function.
def build_MSE_loss(input_res=None):
    mse = tf.keras.losses.MeanSquaredError()

    def MSE_loss(y_true, y_pred):
        return mse(y_true, y_pred)

    return MSE_loss