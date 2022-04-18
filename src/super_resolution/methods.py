### methods.py ###

import tensorflow as tf
from abc import ABC, abstractmethod

class Method(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train_method(self, feature, label):
        pass


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
    def train_method(self, feature, label):
        with tf.GradientTape as gen_tape, tf.GradientTape as disc_tape:
            generated_image = self.generator.network(feature, training=True)

            real_output = self.discriminator.network(label, training=True)
            fake_output = self.discriminator.network(generated_image, training=True)

            gen_loss = self.generator.loss_function(generated_image, label, fake_output)
            disc_loss = self.discriminator.loss_function(real_output, fake_output)

        gen_gradient = gen_tape.gradient(gen_loss, self.generator.network.trainable_variables)
        disc_gradient = disc_tape.gradient(gen_loss, self.discriminator.network.trainable_variables)

        self.generator.optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(disc_gradient, self.discriminator.trainable_variables))


class SingleNetwork(Method):
    def __init__(self, network = None):
        super().__init__()

        self.network = network

    def set_network(self, network):
        self.network = network

    def train_method(self, feature, label):
        with tf.GradientTape as tape:
            generated_image = self.network.network(feature, training=True)

            loss = self.network.loss_function(generated_image, label)

        gradient = tape.gradient(loss, self.network.network.trainable_variables)

        self.network.optimizer.apply_gradients(zip(gradient, self.network.trainable_variables))