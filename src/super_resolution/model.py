### model.py ###
# This module defines the Model class. This class is a container for:
#   - a network (built with a make_ function)
#   - a loss function (defined with a _loss function)
#   - a tensorflow optimizer
##

from utils import LossRecorder, TColors


class Model():
    def __init__(self, build_function=None, resolution=None, loss_function = None, optimizer = None):
        super().__init__()
        self.resolution = resolution

        self.network = None
        if build_function is not None:
            self.network = build_function(self.resolution)
        self.build_function = build_function

        self.loss_function = loss_function
        self.optimizer = optimizer

        self.loss_recorder = LossRecorder()


    ## setter functions for the class variables
    def set_network(self, build_function):
        self.network = build_function(self.resolution)
        self.build_function = build_function

    def set_resolution(self, resolution):
        self.resolution = resolution

    def set_weights(self, weights):
        self.network.set_weights(weights)

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_recorder(self, loss_recorder):
        self.loss_recorder = loss_recorder

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = True

        # Build Function
        if self.build_function is None:
            print(TColors.NOTE + 'Build Function: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Build Function: ' + TColors.OKGREEN + 'available')
        # Network
        if self.network is None:
            print(TColors.NOTE + 'Network: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Network: ' + TColors.OKGREEN + 'available')
        # Resolution
        if self.resolution is None:
            print(TColors.NOTE + 'Resolution: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Resolution: ' + TColors.OKGREEN + 'available')
        # Loss Function
        if self.loss_function is None:
            print(TColors.NOTE + 'Loss Function: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Loss Function: ' + TColors.OKGREEN + 'available')
        # Optimizer
        if self.optimizer is None:
            print(TColors.NOTE + 'Optimizer: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Optimizer: ' + TColors.OKGREEN + 'available')

        # make python print normal
        print(TColors.ENDC)
        
        return status_ok

    # This function is used to create the ABOUT.md file it returns the network name
    def get_info(self):
        # get the name of the loss function
        lf_name = self.loss_function.__name__
        lf_name = lf_name.replace('_loss', '')

        # get the learning rate
        lr = self.optimizer.lr.numpy()

        return self.network.name, lf_name, lr

    # this function saves all class variables into a directory
    def save_variables(self):
        return {
            'weights': self.network.get_weights(),
            'build_function': self.build_function,
            'resolution': self.resolution,
            'loss_function': self.loss_function,
            'optimizer': self.optimizer,
            'loss_recorder': self.loss_recorder
        }

    # this function loads all the given values into class variables
    def load_variables(self, variables):
        self.set_resolution(variables['resolution'])

        self.set_network(variables['build_function'])
        self.network.set_weights(variables['weights'])

        self.set_loss_function(variables['loss_function'])
        self.set_optimizer(variables['optimizer'])
        self.set_loss_recorder(variables['loss_recorder'])