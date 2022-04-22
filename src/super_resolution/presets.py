### presets.py ###

import tensorflow as tf

from model import Model
from network import architectures as arch
from network import loss_functions as lf

import upsampling, methods


def build_SRGAN():
    INPUT_RES = 32
    OUTPUT_RES = 512

    generator = Model(
        network=arch.build_ResNet(),
        loss_function=lf.SRGAN_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    discriminator = Model(
        network=arch.build_SRGAN_disc(),
        loss_function=lf.disc_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    method = methods.AdversarialNetwork(
        generator=generator,
        discriminator=discriminator
    )

    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method,
        name = 'SRGAN'
    )

    return framework


def build_SRResNet():
    INPUT_RES = 32
    OUTPUT_RES = 512

    model = Model(
        network=arch.build_ResNet(),
        loss_function=lf.SRGAN_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    method = methods.SingleNetwork(model=model)

    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method,
        name = 'SRResNet'
    )

    return framework


def build_SRDemo():
    INPUT_RES = 32
    OUTPUT_RES = 128

    model = Model(
        network=arch.build_Demo(),
        loss_function=lf.MSE_loss,
        optimizer=tf.keras.optimizers.Adam(1e-4)
    )

    method = methods.SingleNetwork(model=model)

    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method,
        name='demo'
    )

    return framework