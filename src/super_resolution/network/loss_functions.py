### loss_functions.py ###
# In this module all the loss/cost functions are defined. Every
# function in the project with a _loss suffix is a loss function.
##

import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras import Model

# This loss function will return the sum of a VGG loss
# function and the gen_loss() function. It is used for the SRGAN preset.
def SRGAN_loss(y_true, y_pred, y_disc):
    return SRResNet_loss(y_true, y_pred) + gen_loss(y_disc)

# The loss of a VGG loss function (content loss)
# (with help from: https://github.com/deepak112/Keras-SRGAN/blob/master/Utils_model.py)

# get shape from y_true
shape = (512, 512, 3)

# build vgg 19 model
vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=shape)
vgg19.trainable = False
# set trainable to False
for l in vgg19.layers:
    l.trainable = False

model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
model.trainable = False


def SRResNet_loss(y_true, y_pred):
    # calculate loss
    return tf.math.reduce_mean(tf.math.square(model(y_true) - model(y_pred)))

# The loss function for the SRGAN generator (adversarial loss)
# (from: https://www.tensorflow.org/tutorials/generative/dcgan)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
def gen_loss(y_disc):
    return cross_entropy(tf.ones_like(y_disc), y_disc)

# The loss function of the SRGAN discriminator.
# (from: https://www.tensorflow.org/tutorials/generative/dcgan)
def disc_loss(y_real, y_fake):
    real_loss = cross_entropy(tf.ones_like(y_real), y_real)
    fake_loss = cross_entropy(tf.zeros_like(y_fake), y_fake)

    return real_loss + fake_loss

# This loss function is packs the default tf MSE into a own function. 
mse = tf.keras.losses.MeanSquaredError()
def MSE_loss(y_true, y_pred):
    return mse(y_true, y_pred)