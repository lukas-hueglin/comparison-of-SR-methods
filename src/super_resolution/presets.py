### presets.py ###
# This module contains all the presets, for building a framework. All
# the functions in the pruject with a build_ prefix returns a framework
##

import tensorflow as tf

from model import Model
from network import architectures as arch
from network import loss_functions as lf
from utils import statistics as stats

import upsampling, methods

# This function builds the unsupervised SRGAN framework
def build_SRGAN():
    INPUT_RES = 32
    OUTPUT_RES = 512

    # build generator
    generator = Model(
        build_function=arch.make_SRResNet,
        resolution=OUTPUT_RES,
        loss_function=lf.SRGAN_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build discriminator
    discriminator = Model(
        build_function=arch.make_SRGAN_disc,
        resolution=OUTPUT_RES,
        loss_function=lf.disc_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build method
    method = methods.AdversarialNetwork(
        generator=generator,
        discriminator=discriminator
    )

    # build framework
    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRGAN'
    )

    return framework

# This function builds a supervised SRGAN with just the SRResNet
def build_SRResNet():
    INPUT_RES = 32
    OUTPUT_RES = 512

    # build the model
    model = Model(
        build_function=arch.make_SRResNet,
        resolution=OUTPUT_RES,
        loss_function=lf.SRGAN_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build the method
    method = methods.SingleNetwork(model=model)

    #build the framework
    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRResNet'
    )

    return framework


# This function builds just a test framework
def build_SRDemo():
    INPUT_RES = 32
    OUTPUT_RES = 128

    #builds the model
    model = Model(
        build_function=arch.make_Demo,
        resolution=OUTPUT_RES,
        loss_function=lf.MSE_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4),
    )

    # builds the method
    method = methods.SingleNetwork(model=model)

    # builds the framework
    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name='SRDemo'
    )

    return framework