### methods.py ###

import tensorflow as tf
from abc import ABC, abstractmethod


class Method(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train_method(self, features, labels):
        return labels

    @abstractmethod
    def generate_images(self, images):
        return images


class AdversarialNetwork(Method):
    def __init__(self, generator = None, discriminator = None):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator

    # (from the tensorflow documentation: https://www.tensorflow.org/tutorials/generative/dcgan)
    def train_method(self, features, labels):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_image = self.generator.network(features, training=True)

            real_output = self.discriminator.network(labels, training=True)
            fake_output = self.discriminator.network(generated_image, training=True)

            gen_loss = self.generator.loss_function(labels, generated_image, fake_output)
            disc_loss = self.discriminator.loss_function(real_output, fake_output)

        gen_gradient = gen_tape.gradient(gen_loss, self.generator.network.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.network.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(gen_gradient, self.generator.network.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(disc_gradient, self.discriminator.network.trainable_variables))

        return generated_image

    def generate_images(self, images):
        return self.generator.network(images, training=False)


class SingleNetwork(Method):
    def __init__(self, model = None):
        super().__init__()

        self.model = model

    def set_network(self, model):
        self.model = model
    
    def train_method(self, features, labels):
        with tf.GradientTape() as tape:
            generated_image = self.model.network(features, training=True)

            loss = self.model.loss_function(labels, generated_image)


        gradient = tape.gradient(loss, self.model.network.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradient, self.model.network.trainable_variables))

        return generated_image

    def generate_images(self, images):
        return self.model.network(images, training=False)