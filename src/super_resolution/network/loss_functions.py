### loss_functions.py ###
# In this module all the loss/cost functions are defined. Every
# function in the project with a _loss suffix is a loss function.
# Every loss function is build with a build_ _loss build function.
##

import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras import Model

#from utils import TColors

import numpy as np


# This loss function will return the sum of a VGG loss
# function and the gen_loss() function. It is used for the SRGAN preset.
def build_SRGAN_loss(input_res=None):
    # build SRResNet_loss
    SRResNet_loss = build_SRResNet_loss(input_res)
    # build gen_loss
    gen_loss = build_gen_loss()

    def SRGAN_loss(y_true, y_pred, y_disc):
        ResNet_loss = SRResNet_loss(y_true, y_pred)
        G_loss = (1e-2)*gen_loss(y_disc)
        return (ResNet_loss + G_loss, (ResNet_loss, G_loss))

    return SRGAN_loss

# This loss function will return the sum of a Fourier loss
# function and the gen_loss() function. It is used for the SRGAN_Fourier preset.
def build_SRGAN_Fourier_loss(input_res=None):
    # build SRResNet_loss
    SRResNet_loss = build_SRResNet_loss(input_res)
    # build Fourier_loss
    Fourier_loss = build_Fourier_loss(input_res)
    # MSE loss
    MSE_loss = build_MSE_loss()
    # build gen_loss
    gen_loss = build_gen_loss()

    def SRGAN_Fourier_loss(y_true, y_pred, y_disc):
        ResNet_loss = SRResNet_loss(y_true, y_pred)
        F_loss = 5*(1e2)*Fourier_loss(y_true, y_pred)
        #M_loss = 0*MSE_loss(y_true, y_pred)
        G_loss = (1e-2)*gen_loss(y_disc)
        return (F_loss + ResNet_loss + G_loss, (F_loss, ResNet_loss, G_loss))

    return SRGAN_Fourier_loss

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
        return MSE_loss(model(y_true)/12.75, model(y_pred)/12.75)

    return SRResNet_loss


def build_Fourier_loss(input_res=None):
    # build mse
    MSE_loss = build_MSE_loss()

    def fourier(c):
        transform = tf.signal.fft2d(c)
        return tf.signal.fftshift(transform)

    def inverse_fourier(c):
        shifted = tf.signal.ifftshift(c)
        return tf.signal.ifft2d(shifted)

    def mask(c, b1, b2):
        # create pixel bounds
        pb1, pb2 = int(input_res * b1 / 2), int(input_res * b2 / 2)

        middle = int(input_res / 2)

        # create a black mask
        mask = np.zeros(c.shape)

        # draw the parts, which should stay and the parts which should go
        mask[:, middle - pb2:middle + pb2, middle - pb2:middle + pb2] = 1
        mask[:, middle - pb1:middle + pb1, middle - pb1:middle + pb1] = 0

        return c * tf.cast(tf.convert_to_tensor(mask), tf.complex64)

    def analyze(img, freq_bounds):
        # split channels and convert to complex tensors
        r, g, b = tf.split(tf.cast(img, tf.complex64), num_or_size_splits=3, axis=3)
        r, g, b = tf.reshape(r, r.shape[:-1]), tf.reshape(g, g.shape[:-1]), tf.reshape(b, b.shape[:-1])

        # define return container
        imgs = []

        # perform fourier transform
        (fr, fg, fb) = [fourier(c) for c in (r, g, b)]

        for bound in range(len(freq_bounds)-1):
            # mask frequencies
            bound1 = freq_bounds[bound]
            bound2 = freq_bounds[bound+1]

            (mfr, mfg, mfb) = [mask(c, bound1, bound2) for c in (fr, fg, fb)]

            # perform inverse fourier transform
            (imfr, imfg, imfb) = [tf.reshape(tf.math.abs(inverse_fourier(c)), (-1, input_res, input_res, 1)) for c in (mfr, mfg, mfb)]

            # merge channels
            imgs.append(tf.concat([imfr, imfg, imfb], 3))

        return imgs


    def Fourier_loss(y_true, y_pred):
        # params
        FREQ_BOUNDS = [0, 0.02, 0.15, 0.4,  1]
        #FREQ_WEIGHTS = [np.exp(-((epoch-1)/2)+1.5)+1, epoch/10+1, epoch/8+1, np.min([np.exp((epoch-1)/10-3)+1, 10])]
        FREQ_WEIGHTS = [1/20, 4/20, 7/20, 8/20]

        # analyse images
        images_in = tf.concat([tf.cast((y_true+1)/2, tf.float32), tf.maximum(tf.cast((y_pred+1)/2, tf.float32), 0)], 0)
        images_out = analyze(images_in, FREQ_BOUNDS)

        true_analyzed, pred_analyzed = tf.split(images_out, num_or_size_splits=2, axis=1)

        loss = 0
        # get mse error between images
        for i in range(len(FREQ_WEIGHTS)):
            loss += MSE_loss(true_analyzed[i], pred_analyzed[i]) * FREQ_WEIGHTS[i]

        return (loss / np.sum(FREQ_WEIGHTS))

    return Fourier_loss

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
        loss = real_loss + fake_loss

        return loss, None

    return disc_loss

# This loss function is packs the default tf MSE into a own function.
def build_MSE_loss(input_res=None):
    mse = tf.keras.losses.MeanSquaredError()

    def MSE_loss(y_true, y_pred):
        return mse(y_true, y_pred)

    return MSE_loss