### upsampling.py ###
# In this module the Framework classes and it's child classes are defined. In addition
# the module houses all the upsampling functions used in this project. The Framework
# class connects the upsampling with the training of the networks. It also acts as a
# Container of the whole super-resolution procedure.
##

import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt

import os
from abc import ABC, abstractmethod

from utils import TColors, StatsRecorder


## upsampling functions ##
def bicubic(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.BICUBIC)

def bilinear(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.BICUBIC)

def lanczos(img, res):
    return tf.image.resize(img, (res, res), method=tf.image.ResizeMethod.LANCZOS3)

def none(img, res):
    return img

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

        # Notes for the About.md file
        self.notes = '*Placeholder*\n'

    ## setter functions for the class variables
    def set_resolutions(self, input_res, output_res):
        self.input_res = input_res
        self.output_res = output_res

    def set_stats_recorder(self, stats_recorder):
        self.stats_recorder = stats_recorder

    def set_upsample_function(self, upsample_function):
        self.upsample_function = upsample_function

    # empty function for a training step
    @abstractmethod
    def train_step(self, feature, label):
        pass

    # empty function to generate images
    @abstractmethod
    def generate_images(self, images, check=False):
        pass

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    @abstractmethod
    def check_variables(self):
        status_ok = True

        # Input Resolution
        if self.input_res is None:
            print(TColors.NOTE + 'Input Resolution: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Input Resolution: ' + TColors.OKGREEN + 'available')
        # Output Resolution
        if self.output_res is None:
            print(TColors.NOTE + 'Output Resolution: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Output Resolution: ' + TColors.OKGREEN + 'available')
        # Upsample Function
        if self.upsample_function is None:
            print(TColors.NOTE + 'Updample Function: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Updample Function: ' + TColors.OKGREEN + 'available')

        return status_ok

    # counts the epoch counter up one epoch
    @abstractmethod
    def add_epoch(self):
        pass

    # adds the values to the StatsRecorder
    @abstractmethod
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None, train=True):
        pass

    # empty function for plotting all stats
    @abstractmethod
    def plot_and_save_stats(self, path, name, train=True):
        pass

    # plots the time
    def plot_time(self, path, name, train=True):
        if train:
            fig, ax_train = plt.subplots()
            ax_validation = None
        else:
            fig, (ax_train, ax_validation) = plt.subplots(2)
            fig.tight_layout(pad=3.0)

        fig.suptitle(name)
        self.stats_recorder.plot_time(ax_train, ax_validation, train=train)
        
        # save the plot
        fig.savefig(path + '\\time.png', dpi=300, format='png')

    # plots the system load
    def plot_sys_load(self, path, name, train=True):
        fig = plt.figure(constrained_layout=True)
        if train:
            fig_train = fig.subfigures(1)
            fig_validation = None
        else:
            (fig_train, fig_validation) = fig.subfigures(1, 2)
            fig.tight_layout(pad=1.0)

        fig.suptitle(name)
        self.stats_recorder.plot_sys_load(fig_train, fig_validation, train=train)
        
        # save the plot
        fig.savefig(path + '\\system_load.png', dpi=300, format='png')

    # plots the metrics
    def plot_metrics(self, path, name, train=True):
        fig, axs = plt.subplots(len(self.stats_recorder.metric_functions))
        fig.suptitle(name)

        self.stats_recorder.plot_metrics(axs, train=train)
        
        # save the plot
        fig.savefig(path + '\\metrics.png', dpi=300, format='png')

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
        # add 3 - Statistics
        text += '## 3 - Statistics\n\n'

        minutes, seconds = divmod(self.stats_recorder.training['total_time'], 60)
        hours, minutes = divmod(minutes, 60)
        text += 'total train time: ' + "%d *h*, %d *min*, %d *sec*" % (hours, minutes, seconds) + '</br>\n'

        minutes, seconds = divmod(self.stats_recorder.validation['total_time'], 60)
        hours, minutes = divmod(minutes, 60)
        text += 'total validation time: ' + "%d *h*, %d *min*, %d *sec*" % (hours, minutes, seconds) + '\n\n'
        text += 'Validation metrics:\n'
        for i in range(len(self.stats_recorder.metric_functions)):
            text += '>' + (self.stats_recorder.metric_functions[i].__name__).split('_')[0]
            text += ': ' + (f"{np.mean(self.stats_recorder.validation['metrics'][i]):.2f}" if len(self.stats_recorder.validation['metrics'][i]) != 0 else '-') + '</br>\n'
        
        text += '\n'
        # add 4 - Notes
        text += '\n## 4 - Notes\n\n'
        text += self.notes + '\n'

        return text

    # this function saves all class variables into a directory
    @abstractmethod
    def save_variables(self):
        return {
            'input_res': self.input_res,
            'output_res': self.output_res,
            'upsample_function': self.upsample_function,
            'stats_recorder': self.stats_recorder,
            'name': self.name,
            'notes': self.notes
        }

    # this function loads all the given values into class variables
    @abstractmethod
    def load_variables(self, variables):
        self.input_res = variables['input_res']
        self.output_res = variables['output_res']
        self.upsample_function = variables['upsample_function']
        self.stats_recorder = variables['stats_recorder']
        self.name = variables['name']
        self.notes = variables['notes']

    # this function returns the training arguments
    @abstractmethod
    def get_train_args(self):
        pass


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
    def train_step(self, features, labels, in_args):
        # upsampling the image
        upsampled_features = self.upsample_function(features, self.output_res)

        # passing it to the neural network
        generated_image, loss = self.method.train_method(upsampled_features, labels, in_args)

        return generated_image, loss

    # generates images the same way it is trained
    # the check param is true if the function is used within the pipeline's check() function
    def generate_images(self, images, check=False):
        # check if everything is given
        if check:
            assert(self.check_variables())
        
        # get the prediction of the network
        upsampled_images = self.upsample_function(images, self.output_res)

        return self.method.generate_images(upsampled_images, check=check)

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = super().check_variables()

        # Method
        if self.method is None:
            print(TColors.NOTE + 'Method: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Method: ' + TColors.OKGREEN + 'available')

        # make python print normal
        print(TColors.ENDC)

        return status_ok

    # This function is used to create the ABOUT.md file and
    # returns a string with all the information in 2 - Framework
    def get_info(self, class_name=''):
        class_name = __class__.__name__
        return super().get_info(class_name)

    # counts the epoch counter up one epoch
    def add_epoch(self):
        self.method.add_epoch()
        self.stats_recorder.add_epoch()

    # adds the values to the StatsRecorder
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None, train=True):
        # add loss
        self.method.add_loss(loss)
        # add time
        self.stats_recorder.add_time(time, train=train)
        # add sys_load
        self.stats_recorder.add_sys_load(sys_load, train=train)
        # add metrics
        self.stats_recorder.add_metrics(metrics, train=train)

    # plots the stats
    def plot_and_save_stats(self, path, name, train=True):
        os.makedirs(path, exist_ok=True)

        # plot loss
        self.method.plot_loss(path, name)

        # plot time, sys_load and metrics
        self.plot_time(path, name, train=train)
        self.plot_sys_load(path, name, train=train)
        self.plot_metrics(path, name, train=train)

    # this function saves all class variables into a directory
    def save_variables(self):
        return super().save_variables() | {
            'class': self.__class__,
            'method': self.method.save_variables()
        }

    # this function loads all the given values into class variables
    def load_variables(self, variables):
        super().load_variables(variables)

        self.method = variables['method']['class']()
        self.method.load_variables(variables['method'])

    def get_train_args(self):
        return self.method.get_train_args()



# This class scales the images up progressively. The output image
class ProgressiveUpsampling(Framework):
    def __init__(self, input_res=None, output_res=None, upsample_function=None, methods=None, steps = 1, metric_functions=[], name='framework'):
        super().__init__(input_res, output_res, upsample_function, metric_functions, name)

        # There are multiple methods for each resolution
        self.methods = methods
        self.steps = steps

    ## setter functions for the class variables
    def set_methods(self, methods):
        self.methods = methods

    def set_steps(self, steps):
        self.steps = steps
        self.methods = [self.methods[0]] * self.steps


    # This is the training step function. The labels
    # get scaled up and sent to the network self.step times.
    @tf.function
    def train_step(self, features, labels, in_args):
        generated_images = []
        losses = []
        # there are self.steps loops
        for i in range(self.steps):

            # figure out the resolution on which the net works
            res = self.input_res * np.power(2, i)

            # sample the features up and the lables down
            upsampled_features = self.upsample_function(features, res)
            downsampled_labels = lanczos(labels, res*2) # using lanczos because it's a good downsample method

            # train the network
            generated_image, loss = self.methods[i].train_method(upsampled_features, downsampled_labels, in_args[i])

            # append information for the return value
            losses.append(loss)

            # pass the generated image back into the feature for the next loop
            features = (generated_image + 1 ) / 2

        return features, losses

    # generates images the same way it is trained
    # the check param is true if the function is used within the pipeline's check() function
    def generate_images(self, images, check=False):
        # check if everything is given
        if check:
            assert(self.check_variables())

        # there are self.steps loops
        for i in range(self.steps):
            
            # figure out the resolution on which the net works
            res = self.input_res * np.power(2, i)
            upsampled_images = self.upsample_function(images, res)

            # get prediction of the network
            images = self.methods[i].generate_images(upsampled_images, check=check)

        return images

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = super().check_variables()

        # Methods
        if self.methods is None:
            print(TColors.NOTE + 'Methods: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Methods: ' + TColors.OKGREEN + 'available')
        # Steps
        if self.steps is None:
            print(TColors.NOTE + 'Steps: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Steps: ' + TColors.OKGREEN + 'available')

        # make python print normal
        print(TColors.ENDC)

        return status_ok

    # This function is used to create the ABOUT.md file and
    # returns a string with all the information in 2 - Framework
    def get_info(self, class_name=''):
        class_name = __class__.__name__
        return super().get_info(class_name)

    # counts the epoch counter up one epoch
    def add_epoch(self):
        for method in self.methods:
            method.add_epoch()
        self.stats_recorder.add_epoch()

    # adds the values to the StatsRecorder
    def update_stats_recorder(self, loss=None, time=None, sys_load=None, metrics=None, train=True):
        # add loss
        for i in range(self.steps):
            self.methods[i].add_loss(loss=loss[i])

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

    # this function saves all class variables into a directory
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

    # this function loads all the given values into class variables
    def load_variables(self, variables):
        super().load_variables(variables)

        self.stept = variables['steps']

        for m in variables['methods']:
            method = m['class']()
            method.load_variables(m)
            self.methods.append(method)

    def get_train_args(self):
        loss_list = []

        for i in range(self.steps):
            loss_list.append(self.methods[i].get_train_args())

        return tf.convert_to_tensor(loss_list)