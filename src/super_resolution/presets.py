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
# like in the original SRGAN paper
def build_SRGAN():
    INPUT_RES = 128
    OUTPUT_RES = 512

    # build generator
    generator = Model(
        arch_build_function=arch.make_SRResNet_4x,
        loss_build_function=lf.build_SRGAN_loss,
        resolutions=(INPUT_RES, OUTPUT_RES),
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build discriminator
    discriminator = Model(
        arch_build_function=arch.make_SRGAN_disc,
        loss_build_function=lf.build_disc_loss,
        resolutions=(OUTPUT_RES, OUTPUT_RES),
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
        upsample_function=upsampling.none,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRGAN'
    )

    return framework

# This function builds the unsupervised SRGAN framework,
# but the training of the discriminator is limited
def build_SRGAN_Limited():
    INPUT_RES = 128
    OUTPUT_RES = 512

    # build generator
    generator = Model(
        arch_build_function=arch.make_SRResNet_4x,
        loss_build_function=lf.build_SRGAN_Limited_loss,
        resolutions=(INPUT_RES, OUTPUT_RES),
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build discriminator
    discriminator = Model(
        arch_build_function=arch.make_SRGAN_disc,
        loss_build_function=lf.build_disc_loss,
        resolutions=(OUTPUT_RES, OUTPUT_RES),
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build method
    method = methods.LimitedAdversarialNetwork(
        generator=generator,
        discriminator=discriminator
    )

    # build framework
    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.none,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRGAN_Limited'
    )

    return framework


# This function builds the unsupervised SRGAN framework,
# but the training of the discriminator is limited
def build_SRGAN_Limited_Progressive():
    INPUT_RES = 32
    OUTPUT_RES = 128
    STEPS = 2
    
    # empty methods list
    method_list = []

    for i in range(STEPS):
        # build generator
        generator = Model(
            arch_build_function=arch.make_SRResNet_2x,
            loss_build_function=lf.build_SRGAN_loss,
            resolutions=(int(INPUT_RES*(i+1)), int(OUTPUT_RES/(2-i))),
            optimizer=tf.keras.optimizers.Adam(1e-4)
        )

        # build discriminator
        discriminator = Model(
            arch_build_function=arch.make_SRGAN_disc,
            loss_build_function=lf.build_disc_loss,
            resolutions=(int(OUTPUT_RES/(2-i)), int(OUTPUT_RES/(2-i))),
            optimizer=tf.keras.optimizers.Adam(1e-4)
        )

        # build method
        method = methods.LimitedAdversarialNetwork(
            generator=generator,
            discriminator=discriminator
        )

        # add the method
        method_list.append(method)

    # build framework
    framework = upsampling.ProgressiveUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.none,
        methods=method_list,
        steps=STEPS,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRGAN_Limited_Progressive'
    )

    return framework


def build_SRGAN_Fourier():
    INPUT_RES = 128
    OUTPUT_RES = 512

    # build generator
    generator = Model(
        arch_build_function=arch.make_SRResNet_4x,
        loss_build_function=lf.build_SRGAN_Fourier_loss,
        resolutions=(INPUT_RES, OUTPUT_RES),
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build discriminator
    discriminator = Model(
        arch_build_function=arch.make_SRGAN_disc,
        loss_build_function=lf.build_disc_loss,
        resolutions=(OUTPUT_RES, OUTPUT_RES),
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
        upsample_function=upsampling.none,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRGAN_Fourier'
    )

    return framework


# This function builds a supervised SRGAN with just the SRResNet
def build_SRResNet():
    INPUT_RES = 128
    OUTPUT_RES = 512

    # build the model
    model = Model(
        arch_build_function=arch.make_SRResNet_4x,
        loss_build_function=lf.build_MSE_loss,
        resolutions=(INPUT_RES, OUTPUT_RES),
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build the method
    method = methods.SingleNetwork(model=model)

    #build the framework
    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.none,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRResNet'
    )

    return framework

# This function builds a supervised SRGAN with just the SRResNet
def build_SRResNet_Fourier():
    INPUT_RES = 128
    OUTPUT_RES = 512

    # build the model
    model = Model(
        arch_build_function=arch.make_SRResNet_4x,
        loss_build_function=lf.build_Fourier_loss,
        resolutions=(INPUT_RES, OUTPUT_RES),
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    # build the method
    method = methods.SingleNetwork(model=model)

    #build the framework
    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.none,
        method=method,
        metric_functions=[stats.PSNR_metric, stats.SSIM_metric],
        name = 'SRResNet_Fourier'
    )

    return framework


# This function builds just a test framework
def build_SRDemo():
    INPUT_RES = 32
    OUTPUT_RES = 128

    #builds the model
    model = Model(
        arch_build_function=arch.make_Demo,
        loss_build_function=lf.build_MSE_loss,
        resolutions=(OUTPUT_RES, OUTPUT_RES),
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