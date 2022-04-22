### methods.py ###
# In this module, the Method class and it's child classes are defined.
# The purpose of these classes is, to make it possible to easily implement different types of super-resolution methods.
# for example:
#   - a GAN needs two models (Model class)
#   - a normal unsupervised method needs just one model.
# With the Method child classes the user can add new Method types as long he defines a train_method(x, y) function.
##

import tensorflow as tf
from abc import ABC, abstractmethod


# Abstract Method class. It is practically empty and just
# requests a train_method(x, y) (and a generate_images(x)) function.
class Method(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train_method(self, features, labels):
        return labels

    @abstractmethod
    def generate_images(self, images):
        return images


# The AdversarialNetwork class describes a method with a generator and a discriminator.
class AdversarialNetwork(Method):
    def __init__(self, generator = None, discriminator = None):
        super().__init__()

        self.generator = generator # the generator model
        self.discriminator = discriminator # the discriminator model

    ## setter functions for the class variables
    def set_generator(self, generator):
        self.generator = generator

    def set_discriminator(self, discriminator):
        self.discriminator = discriminator


    # The train_method(x, y) function trains the generator
    # and the discriminator with the typical GAN procedure.
    # (from the tensorflow documentation: https://www.tensorflow.org/tutorials/generative/dcgan)
    def train_method(self, features, labels):
        # The GradientTapes watch the transformation of the network parameters
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

            # predition by the generator
            generated_image = self.generator.network(features, training=True)

            # prediction by the discriminator given a real image
            # (real_output) and given the generated image of the generator
            real_output = self.discriminator.network(labels, training=True)
            fake_output = self.discriminator.network(generated_image, training=True)

            # calculate the loss
            gen_loss = self.generator.loss_function(labels, generated_image, fake_output)
            disc_loss = self.discriminator.loss_function(real_output, fake_output)

        # calculate the gradient of generator and discriminator
        gen_gradient = gen_tape.gradient(gen_loss, self.generator.network.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.network.trainable_variables)

        # adjust the parameters with backpropagation
        self.generator.optimizer.apply_gradients(zip(gen_gradient, self.generator.network.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(disc_gradient, self.discriminator.network.trainable_variables))

        return generated_image

    # returns the generated image of the generator
    def generate_images(self, images):
        return self.generator.network(images, training=False)


# The SingleNetwork class describes a method with just one model.
class SingleNetwork(Method):
    def __init__(self, model = None):
        super().__init__()
        self.model = model

    # set function for the model
    def set_network(self, model):
        self.model = model
    

    # The train_method(x, y) function trains
    # the model,by minimizing the loss of the predicted image.
    def train_method(self, features, labels):
        # The GradientTape watches the transformation of the network parameters
        with tf.GradientTape() as tape:

            # prediction by the network of the model
            generated_image = self.model.network(features, training=True)

            # calculate the loss
            loss = self.model.loss_function(labels, generated_image)

        # calculate the gradient and and adjusting the network parameters
        gradient = tape.gradient(loss, self.model.network.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradient, self.model.network.trainable_variables))

        return generated_image

    # returns the generated image og the network
    def generate_images(self, images):
        return self.model.network(images, training=False)