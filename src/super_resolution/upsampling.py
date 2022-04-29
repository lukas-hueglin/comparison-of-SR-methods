### upsampling.py ###
# In this module the Framework classes and it's child classes are defined. In addition
# the module houses all the upsampling functions used in this project. The Framework
# class connects the upsampling with the training of the networks. It also acts as a
# Container of the whole super-resolution procedure.
##

import tensorflow as tf

import os

import numpy as np
from abc import ABC, abstractmethod

from utils import TColors


## upsampling functions ##
def bicubic(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.BICUBIC)

def bilinear(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.BICUBIC)

def lanczos(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.LANCZOS3)


## Framework classes ##
class Framework(ABC):
    def __init__(self, input_res = None, output_res = None, upsample_function = None, name = 'framework'):
        # The input and output resolution of the images
        self.input_res = input_res
        self.output_res = output_res

        # The upsampling type/function
        self.upsample_function = upsample_function
        # The name of the "algorithm" used for saving the models.
        self.name = name

        # REMINDER: the method itself is only specified in the child classes

        self.check_resolution()

    ## setter functions for the class variables
    def set_resolutions(self, input_res, output_res):
        self.input_res = input_res
        self.output_res = output_res
        self.check_resolution()

    def set_upsample_function(self, upsample_function):
        self.upsample_function = upsample_function
    
    # This functions checks if the resolutions are all a power of 2
    def check_resolution(self):
        if self.input_res is not None and self.output_res is not None:
            # make it a float so the is_integer functions is accessible
            q = float(np.log2(self.output_res/self.input_res))

            if q.is_integer() is False:
                print(TColors.WARNING + 'The input and output resolutions are not a power of 2!' + TColors.ENDC)

        else:
            print(TColors.WARNING + 'The resolutions are not specified!' + TColors.ENDC)

    # empty function for a training step
    @abstractmethod
    def train_step(self, feature, label):
        pass

    # empty function to generate images
    @abstractmethod
    def generate_images(self, images):
        pass

    # This function is used to create the ABOUT.md file and
    # returns a string with all the information in 2 - Framework
    @abstractmethod
    def get_info(self, class_name=''):
        # title
        text = '# ' + self.name + '\n\n'
        # get text from method
        text += self.method.get_info()
        # add 2 - Framework
        text += '## 2 - Framework\n\n'
        text += 'Framework: *' + class_name + '* </br>\n'
        text += 'Upsample method: *' + self.upsample_function.__name__ + '*\n\n'
        text += 'Input resolution: ' + str(self.input_res) + '*px* </br>\n'
        text += 'Output resolution: ' + str(self.output_res) + '*px*\n\n'

        return text

    @abstractmethod
    # adds the values to the StatsRecorder
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None):
        pass


# This class scales the input images
# up first and passes them aferwards to the network
class PreUpsampling(Framework):
    def __init__(self, input_res=None, output_res=None, upsample_function=None, method=None, name='framework'):
        super().__init__(input_res, output_res, upsample_function, name)

        # added the method variable
        self.method = method

    # set function for the method
    def set_method(self, method):
        self.method = method

    # This is the training step function. The labels
    # get scaled up first and are sent to the network later
    @tf.function
    def train_step(self, features, labels):
        # upsampling the image
        upsampled_features = self.upsample_function(features, self.output_res)

        # passing it to the neural network
        generated_image, loss = self.method.train_method(upsampled_features, labels)

        return generated_image, loss

    # generates images the same way it is trained
    def generate_images(self, images):
        # get the prediction of the network
        upsampled_images = self.upsample_function(images, self.output_res)

        return self.method.generate_images(upsampled_images)

    # This function is used to create the ABOUT.md file and
    # returns a string with all the information in 2 - Framework
    def get_info(self, class_name=''):
        class_name = __class__.__name__
        return super().get_info(class_name)

    # adds the values to the StatsRecorder
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None):
        self.method.update_stats_recorder(loss=loss, time=time, sys_load=sys_load, metrics=metrics)
        
    # plots the stats
    def plot_and_save_stats(self, path, epochs, name):
        os.makedirs(path)

        # plot
        #self.method.plot_loss(path, epochs, name)
        self.method.plot_time(path, epochs, name)
        self.method.plot_sys_load(path, epochs, name)
        

# This class scales the images up progressively. The output image
class ProgressiveUpsampling(Framework):
    def __init__(self, input_res=None, output_res=None, upsample_function=None, methods=None, steps = 1, name='framework'):
        super().__init__(input_res, output_res, upsample_function, name)

        # There are multiple methods for each resolution
        self.methods = methods
        self.steps = steps

    ## setter functions for the class variables
    def set_methods(self, methods):
        self.methods = methods

    def set_steps(self, steps):
        self.steps = steps
        self.methods = [self.methods[0]] * self.steps
        self.check_resolution()

    # This functions overridden function checks also if the steps are correct
    def check_resolution(self):
        if self.methods != None:
            q = super().check_resolution()

            if self.steps != q-1 or len(self.methods) != self.steps:
                print(TColors.WARNING + 'The steps do not match!' + TColors.ENDC)

        else:
            print(TColors.WARNING + 'The methods are not specified!' + TColors.ENDC)


    # This is the training step function. The labels
    # get scaled up and sent to the network self.step times.
    @tf.function
    def train_step(self, features, labels):
        generated_images = []
        losses = []
        # there are self.steps loops
        for i in range(self.steps):

            # figure out the resolution on which the net works
            res = self.input_res * np.power(2, i)

            # sample the features up and the lables down
            upsampled_features = self.upsample_function(features, res)
            downsampled_labels = lanczos(labels, res) # using lanczos because it's a good downsample method

            # train the network
            generated_image, loss = self.method[i].train_method(upsampled_features, downsampled_labels)

            # append information for the return value
            generated_images.append(generated_image)
            losses.append(loss)

            # pass the generated image back into the feature for the next loop
            features = generated_image

        return generated_images, losses

    # generates images the same way it is trained
    def generate_images(self, images):
        # there are self.steps loops
        for i in range(self.steps):
            
            # figure out the resolution on which the net works
            res = self.input_res * np.power(2, i)
            upsampled_images = self.upsample_function(images, res)

            # get prediction of the network
            images = self.method[i].generate_images(upsampled_images)

        return images

    # This function is used to create the ABOUT.md file and
    # returns a string with all the information in 2 - Framework
    def get_info(self, class_name=''):
        class_name = __class__.__name__
        return super().get_info(class_name)

    # adds the values to the StatsRecorder
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None):
        # add time and sys_load just to the first one
        self.methods[0].update_stats_recorder(loss=loss[0], time=time, sys_load=sys_load, metrics=metrics[i])

        for i in range(1, self.steps):
            self.methods[i].update_stats_recorder(loss=loss[i], metrics=metrics[i])
        
    # plots the stats
    def plot_and_save_stats(self, path, epochs, name):
        os.makedirs(path)

        # plot
        for method in self.methods:
            method.plot_loss(path, epochs, name)
        self.methods[0].plot_time(path, epochs, name)
        self.methods[0].plot_sys_load(path, epochs, name)
