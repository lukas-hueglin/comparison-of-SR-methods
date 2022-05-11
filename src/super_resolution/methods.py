### methods.py ###
# In this module, the Method class and it's child classes are defined.
# The purpose of these classes is, to make it possible to easily implement different types of super-resolution methods.
# for example:
#   - a GAN needs two models (Model class)
#   - a normal unsupervised method needs just one model.
# With the Method child classes the user can add new Method types as long he defines a train_method(x, y) function.
##

import tensorflow as tf

import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from model import Model


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

    @abstractmethod
    def add_loss(self, loss):
        pass

    @abstractmethod
    def add_epoch(self):
        pass

    @abstractmethod
    def plot_loss(self, path, epochs, name):
        pass

    @abstractmethod
    def get_info(self):
        pass

    @abstractmethod
    def save_variables(self):
        pass

    @abstractmethod
    def load_variables(self, variables):
        pass


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
            generated_images = self.generator.network(features, training=True)

            # prediction by the discriminator given a real image
            # (real_output) and given the generated image of the generator
            real_output = self.discriminator.network(labels, training=True)
            fake_output = self.discriminator.network(generated_images, training=True)

            # calculate the loss
            gen_loss = self.generator.loss_function(labels, generated_images, fake_output)
            disc_loss = self.discriminator.loss_function(real_output, fake_output)

        # calculate the gradient of generator and discriminator
        gen_gradient = gen_tape.gradient(gen_loss, self.generator.network.trainable_variables)
        disc_gradient = disc_tape.gradient(disc_loss, self.discriminator.network.trainable_variables)

        # adjust the parameters with backpropagation
        self.generator.optimizer.apply_gradients(zip(gen_gradient, self.generator.network.trainable_variables))
        self.discriminator.optimizer.apply_gradients(zip(disc_gradient, self.discriminator.network.trainable_variables))

        # return loss because it can't be accessed in a @tf.function
        return generated_images, (gen_loss, disc_loss)

    # returns the generated image of the generator
    def generate_images(self, images):
        return self.generator.network(images, training=False)


    # adds the values to the StatsRecorder
    def add_loss(self, loss):
        # unpack loss
        gen_loss, disc_loss = loss

        # add loss
        self.generator.loss_recorder.add_loss(gen_loss)
        self.discriminator.loss_recorder.add_loss(disc_loss)

    def add_loss(self):
        self.generator.loss_recorder.add_epoch()
        self.discriminator.loss_function.add_epoch()

    # plots the loss
    def plot_loss(self, path, name):
        fig, (gen, disc) = plt.subplots(2)

        self.generator.loss_recorder.plot_loss(gen)
        self.discriminator.loss_recorder.plot_loss(disc)

        # set titles
        fig.suptitle(name)
        gen.title.set_text('Generator')
        disc.title.set_text('Discriminator')
        
        # save the plot
        fig.savefig(path + '\\loss.png', dpi=300, format='png')

    # This function is used to create the ABOUT.md file and
    # returns a string with all the information in 1 - Method
    def get_info(self):
        # helper function for getting the model info
        def get_model_text(model):
            net, lf, lr = model.get_info()
            t = '>Network: *' + net + '* </br>\n'
            t += '>Loss Function: *' + lf + '* </br>\n'
            t += '>Learning Rate: *' + str(lr) + '* </br>\n\n'
            return t

        # add base text
        text = '![](https://drive.google.com/uc?export=view&id=1kOSUF1jnPmSTR27yNlODdlUa6nARYGmy)\n\n'
        text += '## 1 - Method\n\n'
        text += 'Method: *'
        # add class name
        text += __class__.__name__+'*\n\n'
        # add generator and discriminator
        text += 'Generator:\n' + get_model_text(self.generator)
        text += 'Discriminator:\n' + get_model_text(self.discriminator)

        return text

    def save_variables(self):
        return {
            'class': self.__class__,
            'generator': self.generator.save_variables(),
            'discriminator': self.discriminator.save_variables(),
        }

    def load_variables(self, variables):
        self.generator = Model()
        self.generator.load_variables(variables['generator'])
        self.discriminator = Model()
        self.discriminator.load_variables(variables['discriminator'])


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
            generated_images = self.model.network(features, training=True)

            # calculate the loss
            loss = self.model.loss_function(labels, generated_images)

        # calculate the gradient and and adjusting the network parameters
        gradient = tape.gradient(loss, self.model.network.trainable_variables)
        self.model.optimizer.apply_gradients(zip(gradient, self.model.network.trainable_variables))
        
        # return loss because it can't be accessed in a @tf.function
        return generated_images, loss

    # returns the generated image og the network
    def generate_images(self, images):
        return self.model.network(images, training=False)


    # adds the values to the StatsRecorder
    def add_loss(self, loss):
        # add loss
        self.model.loss_recorder.add_loss(loss)

    def add_epoch(self):
        self.model.loss_recorder.add_epoch()

    # plots the loss
    def plot_loss(self, path, name):
        fig, ax = plt.subplots()
        fig.suptitle(name)

        self.model.loss_recorder.plot_loss(ax)
        
        # save the plot
        fig.savefig(path + '\\loss.png', dpi=300, format='png')

    # This function is used to create the ABOUT.md file and
    # returns a string with all the information in 1 - Method
    def get_info(self):
        # helper function for getting the model info
        def get_model_text(model):
            net, lf, lr = model.get_info()
            t = '>Network: *' + net + '* </br>\n'
            t += '>Loss Function: *' + lf + '* </br>\n'
            t += '>Learning Rate: *' + str(lr) + '* </br>\n\n'
            return t

        # add base text
        text = '![](https://drive.google.com/uc?export=view&id=1kOSUF1jnPmSTR27yNlODdlUa6nARYGmy)\n\n'
        text += '## 1 - Method\n\n'
        text += 'Method: *'
        # add class name
        text += __class__.__name__+'*\n\n'
        # add model
        text += 'Model:\n' + get_model_text(self.model)
        
        return text

    def save_variables(self):
        return {
            'class': self.__class__,
            'model': self.model.save_variables(),
        }

    def load_variables(self, variables):
        self.model = Model()
        self.model.load_variables(variables['model'])