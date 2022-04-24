### methods.py ###
# In this module, the Method class and it's child classes are defined.
# The purpose of these classes is, to make it possible to easily implement different types of super-resolution methods.
# for example:
#   - a GAN needs two models (Model class)
#   - a normal unsupervised method needs just one model.
# With the Method child classes the user can add new Method types as long he defines a train_method(x, y) function.
##

import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

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

    @abstractmethod
    def update_stats_recorder(self, loss):
        pass

    @abstractmethod
    def save_stats(self, path, epochs, name):
        pass

    @abstractmethod
    def get_info(self):
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


    # adds the loss to the StatsRecorder
    def update_stats_recorder(self, loss):
        # unpack loss
        gen_loss, disc_loss = loss

        # add loss to StatsRecorder
        self.generator.stats_recorder.add_loss(gen_loss)
        self.discriminator.stats_recorder.add_loss(disc_loss)

    # plots and saves the stats
    def save_stats(self, path, epochs, name):
        fig, (gen, disc) = plt.subplots(2)
        fig.suptitle(name)

        # get y - axis
        y_gen = self.generator.stats_recorder.tracked_loss
        y_disc = self.discriminator.stats_recorder.tracked_loss

        # make x - axis
        x = np.linspace(0, epochs, num=len(y_gen))
        
        # set labels
        gen.set(xlabel='epochs', ylabel='loss')
        disc.set(xlabel='epochs', ylabel='loss')
        gen.label_outer()
        disc.label_outer()

        # plot
        gen.plot(x, y_gen, linewidth=1)
        disc.plot(x, y_disc, linewidth=1)
        plt.show()
        
        # save the plot
        os.makedirs(path, exist_ok=True)
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


    # adds the loss to the StatsRecorder
    def update_stats_recorder(self, loss):
        self.model.stats_recorder.add_loss(loss)

    # plots the stats
    def save_stats(self, path, epochs, name):
        fig, ax = plt.subplots()
        fig.suptitle(name)
        
        # get y - axis
        y = self.model.stats_recorder.tracked_loss

        # make x - axis
        x = np.linspace(0, epochs, num=len(y))
        
        # set labels
        ax.set(xlabel='epochs', ylabel='loss')
        ax.label_outer()

        # plot
        ax.plot(x, y, linewidth=1)
        plt.show()
        
        # save the plot
        os.makedirs(path, exist_ok=True)
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