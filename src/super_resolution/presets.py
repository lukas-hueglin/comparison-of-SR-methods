### presets.py ###

from keras.optimizers import Adam
from super_resolution import upsampling, methods, Model
from super_resolution.network import architectures as arch
from super_resolution.network import loss_functions as lf


def build_SRGAN():
    INPUT_RES = 32
    OUTPUT_RES = 512

    generator = Model(
        network=arch.build_ResNet(),
        loss_function=lf.SRGAN_loss(),
        optimizer=Adam(1e-4)
    )

    discriminator = Model(
        network=arch.build_SRGAN_disc(),
        loss_function=lf.disc_loss(),
        optimizer=Adam(1e-4)
    )

    method = methods.AdversarialNetwork(
        generator=generator,
        discriminator=discriminator
    )

    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method
    )

    return framework


def build_SRResNet():
    INPUT_RES = 32
    OUTPUT_RES = 512

    network = Model(
        network=arch.build_ResNet(),
        loss_function=lf.SRGAN_loss(),
        optimizer=Adam(1e-4)
    )

    method = methods.SingleNetwork(generator=network)

    framework = upsampling.PreUpsampling(
        input_res=INPUT_RES,
        output_res=OUTPUT_RES,
        upsample_function=upsampling.bicubic,
        method=method
    )

    return framework