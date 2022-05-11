### upsampling.py ###
# In this module the Framework classes and it's child classes are defined. In addition
# the module houses all the upsampling functions used in this project. The Framework
# class connects the upsampling with the training of the networks. It also acts as a
# Container of the whole super-resolution procedure.
##

import tensorflow as tf

import os
import matplotlib.pyplot as plt

import numpy as np
from abc import ABC, abstractmethod

from utils import TColors, StatsRecorder


## upsampling functions ##
def bicubic(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.BICUBIC)

def bilinear(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.BICUBIC)

def lanczos(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.LANCZOS3)


## Framework classes ##
class Framework(ABC):
    def __init__(self, input_res = None, output_res = None, upsample_function = None, metric_functions=[], name = 'framework'):
        # The input and output resolution of the images
        self.input_res = input_res
        self.output_res = output_res

        # The upsampling type/function
        self.upsample_function = upsample_function

        self.stats_recorder = StatsRecorder(metric_functions)

        # The name of the "algorithm" used for saving the models.
        self.name = name

        # REMINDER: the method itself is only specified in the child classes

        self.check_resolution()

    ## setter functions for the class variables
    def set_resolutions(self, input_res, output_res):
        self.input_res = input_res
        self.output_res = output_res
        self.check_resolution()

    def set_stats_recorder(self, stats_recorder):
        self.stats_recorder = stats_recorder

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
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None, train=True):
        pass

    @abstractmethod
    def plot_and_save_stats(self, path, name, train=True):
        pass

    @abstractmethod
    def add_epoch(self):
        pass

    def plot_time(self, path, name, train=True):
        if train:
            fig, ax_train = plt.subplots()
            ax_validation = None
        else:
            fig, (ax_train, ax_validation) = plt.subplots(2)

        fig.suptitle(name)
        self.stats_recorder.plot_time(ax_train, ax_validation, train=train)
        
        # save the plot
        fig.savefig(path + '\\time.png', dpi=300, format='png')

    def plot_sys_load(self, path, name, train=True):
        if train:
            fig, ax_train = plt.subplots()
            ax_validation = None
        else:
            fig, (ax_train, ax_validation) = plt.subplots(2)

        fig.suptitle(name)
        self.stats_recorder.plot_sys_load(ax_train, ax_validation, train=train)
        
        # save the plot
        fig.savefig(path + '\\system_load.png', dpi=300, format='png')

    # plots the metrics
    def plot_metrics(self, path, name, train=True):
        fig, axs = plt.subplots(len(self.stats_recorder.metric_functions))
        fig.suptitle(name)

        self.stats_recorder.plot_metrics(axs, train=train)
        
        # save the plot
        fig.savefig(path + '\\metrics.png', dpi=300, format='png')

    @abstractmethod
    def save_variables(self):
        return {
            'input_res': self.input_res,
            'output_res': self.output_res,
            'upsample_function': self.upsample_function,
            'stats_recorder': self.stats_recorder,
            'name': self.name
        }

    @abstractmethod
    def load_variables(self, variables):
        self.input_res = variables['input_res']
        self.output_res = variables['output_res']
        self.upsample_function = variables['upsample_function']
        self.stats_recorder = variables['stats_recorder']
        self.name = variables['name']


# This class scales the input images
# up first and passes them aferwards to the network
class PreUpsampling(Framework):
    def __init__(self, input_res=None, output_res=None, upsample_function=None, method=None, metric_functions=[], name='framework'):
        super().__init__(input_res, output_res, upsample_function, metric_functions, name)

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
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None, train=True):
        # update loss
        if train:
            self.method.add_loss(loss)

        # add time
        self.stats_recorder.add_time(time, train=train)
        # add sys_load
        self.stats_recorder.add_sys_load(sys_load, train=train)
        # add metrics
        self.stats_recorder.add_metrics(metrics, train=train)

    def add_epoch(self):
        self.method.add_epoch()
        self.stats_recorder.add_epoch()

    # plots the stats
    def plot_and_save_stats(self, path, name, train=True):
        os.makedirs(path, exist_ok=True)

        # plot loss
        self.method.plot_loss(path, name)

        # plot time, sys_load and metrics
        self.plot_time(path, name, train=train)
        self.plot_sys_load(path, name, train=train)
        self.plot_metrics(path, name, train=train)

    def save_variables(self):
        return super().save_variables() | {
            'class': self.__class__,
            'method': self.method.save_variables()
        }

    def load_variables(self, variables):
        super().load_variables(variables)

        self.method = variables['method']['class']()
        self.method.load_variables(variables['method'])
        

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
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None, train=True):
        # update loss
        if train:
            for i in range(self.steps):
                self.methods[i].update_stats_recorder(loss=loss[i])

        # add time
        self.stats_recorder.add_time(time, train=train)
        # add sys_load
        self.stats_recorder.add_sys_load(sys_load, train=train)
        # add metrics
        self.stats_recorder.add_metrics(metrics, train=train)
        
    # plots the stats
    def plot_and_save_stats(self, path, epochs, name, train=True):
        os.makedirs(path, exist_ok=True)

        # plot loss
        for method in self.methods:
            method.plot_loss(path, epochs, name)

        # plot time, sys_load and metrics
        self.plot_time(path, name, train=train)
        self.plot_sys_load(path, name, train=train)
        self.plot_metrics(path, name, train=train)

    def add_epochs(self):
        for method in self.methods:
            method.add_epoch()
        self.stats_recorder.add_epoch()

    def save_variables(self):
        methods = {}
        for i in range(len(self.methods)):
            methods = methods | {'method_'+str(i): self.methods[i].save_variables()}

        dict = {
            'class': self.__class__,
            'methods': methods,
            'steps': self.steps
        }

        return dict | super().save_variables()

    def load_variables(self, variables):
        super().load_variables(variables)

        self.stept = variables['steps']

        for m in variables['methods']:
            method = m['class']()
            method.load_variables(m)
            self.methods.append(method)
