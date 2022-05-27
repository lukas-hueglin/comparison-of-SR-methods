### loss_functions.py ###
# In this module all the loss/cost functions are defined. Every
# function in the project with a _loss suffix is a loss function.
##

import tensorflow as tf

# This loss function will return the sum of a VGG loss
# function and the gen_loss() function. It is used for the SRGAN preset.
def SRGAN_loss():
    pass

# The loss function for the SRGAN generator.
def gen_loss():
    pass

# The loss function of the SRGAN discriminator.
def disc_loss():
    pass

# This loss function is packs the default tf MSE into a own function. 
mse = tf.keras.losses.MeanSquaredError()
def MSE_loss(y_true, y_pred):
    return mse(y_true, y_pred)