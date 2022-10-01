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
from utils import TColors

import numpy as np


# Abstract Method class. It is practically empty and just
# requests a train_method(x, y) (and a generate_images(x)) function.
class Method(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train_method(self, features, labels):
        return labels

    @abstractmethod
    def generate_images(self, images, check=False):
        return images

    @abstractmethod
    def check_variables(self, verbose=False):
        pass

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

    @abstractmethod
    def get_train_args(self):
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
    def train_method(self, features, labels, in_args):
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
    # the check param is true if the function is used within the pipeline's check() function
    def generate_images(self, images, check=False):
        # check if everything is given
        if check:
            assert(self.check_variables())
            assert(self.generator.check_variables())
            assert(self.discriminator.check_variables())

        # the first time tensorflow generated a output the cuDNN library gets loaded
        if check:
            print(TColors.OKGREEN + 'NVIDIA cuDNN' + TColors.NOTE + ' Version:')

        return self.generator.network(images, training=False)

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = True

        # Generator
        if self.generator is None:
            print(TColors.NOTE + 'Generator: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Generator: ' + TColors.OKGREEN + 'available')
        # Discriminator
        if self.discriminator is None:
            print(TColors.NOTE + 'Discriminator: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Discriminator: ' + TColors.OKGREEN + 'available')

        # make python print normal
        print(TColors.ENDC)

        return status_ok

    # adds the values to the StatsRecorder
    def add_loss(self, loss):
        if loss is not None:
            # unpack loss
            gen_loss, disc_loss = loss

            # add loss
            self.generator.loss_recorder.add_loss(gen_loss)
            self.discriminator.loss_recorder.add_loss(disc_loss)

    # counts the epoch counter up one epoch
    def add_epoch(self):
        self.generator.loss_recorder.add_epoch()
        self.discriminator.loss_recorder.add_epoch()

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

    # this function saves all class variables into a directory
    def save_variables(self):
        return {
            'class': self.__class__,
            'generator': self.generator.save_variables(),
            'discriminator': self.discriminator.save_variables(),
        }

    # this function loads all the given values into class variables
    def load_variables(self, variables):
        self.generator = Model()
        self.generator.load_variables(variables['generator'])
        self.discriminator = Model()
        self.discriminator.load_variables(variables['discriminator'])

    def get_train_args(self):
        return None

# The AdversarialNetwork class describes a method with a generator and a discriminator,
# but the training of the discriminator is limited
class LimitedAdversarialNetwork(AdversarialNetwork):
    def __init__(self, generator = None, discriminator = None):
        super().__init__(generator, discriminator)

    # The train_method(x, y) function trains the generator
    # and the discriminator with the typical GAN procedure.
    # (from the tensorflow documentation: https://www.tensorflow.org/tutorials/generative/dcgan)
    def train_method(self, features, labels, in_args):
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

        # only apply gradient if discriminator is not too good
        optimal_loss = 0.6932 #ln(2)
        if in_args >= optimal_loss:
            self.discriminator.optimizer.apply_gradients(zip(disc_gradient, self.discriminator.network.trainable_variables))

        # return loss because it can't be accessed in a @tf.function
        return generated_images, (gen_loss, disc_loss)

    # this function returns the loss of the discriminator as train arguments
    def get_train_args(self):
        loss = self.discriminator.loss_recorder.loss
        if len(loss) == 0:
            return tf.Variable(0, dtype=tf.float32, trainable=False)
        else:
            return tf.Variable(np.mean(loss[-400:]), dtype=tf.float32, trainable=False)


# The SingleNetwork class describes a method with just one model.
class SingleNetwork(Method):
    def __init__(self, model = None):
        super().__init__()
        self.model = model

    # set function for the model
    def set_model(self, model):
        self.model = model
    
    # The train_method(x, y) function trains
    # the model,by minimizing the loss of the predicted image.
    def train_method(self, features, labels, in_args):
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

    # returns the generated image of the network
    # the check param is true if the function is used within the pipeline's check() function
    def generate_images(self, images, check=False):
        # check if everything is given
        if check:
            assert(self.check_variables())
            assert(self.model.check_variables())

        # the first time tensorflow generated a output the cuDNN library gets loaded
        if check:
            print(TColors.OKGREEN + 'NVIDIA cuDNN' + TColors.NOTE + ' Version:')

        return self.model.network(images, training=False)

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = True

        # Model
        if self.model is None:
            print(TColors.NOTE + 'Model: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Model: ' + TColors.OKGREEN + 'available')

        # make python print normal
        print(TColors.ENDC)
        
        return status_ok

    # adds the values to the StatsRecorder
    def add_loss(self, loss):
        # add loss
        self.model.loss_recorder.add_loss(loss)

    # counts the epoch counter up one epoch
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

    # this function saves all class variables into a directory
    def save_variables(self):
        return {
            'class': self.__class__,
            'model': self.model.save_variables(),
        }

    # this function loads all the given values into class variables
    def load_variables(self, variables):
        self.model = Model()
        self.model.load_variables(variables['model'])

    def get_train_args(self):
        return None