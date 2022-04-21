### loss_functions.py ###

import tensorflow as tf

def SRGAN_loss():
    pass

def gen_loss():
    pass

def disc_loss():
    pass

def MSE_loss(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return mse(y_true, y_pred)