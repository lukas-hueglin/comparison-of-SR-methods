### model.py ###
# This module defines the Model class. This class is a container for:
#   - a network (built with a make_ function)
#   - a loss function (defined with a _loss function)
#   - a tensorflow optimizer
##

from utils import LossRecorder, TColors


class Model():
    def __init__(self, arch_build_function=None, loss_build_function = None, resolutions=(None, None), optimizer = None):
        super().__init__()
        self.input_res, self.output_res = resolutions

        self.network = None
        if arch_build_function is not None:
            self.network = arch_build_function(self.input_res)
        self.arch_build_function = arch_build_function

        self.loss_function = None
        if loss_build_function is not None:
            self.loss_function = loss_build_function(input_res=self.output_res)
        self.loss_build_function = loss_build_function

        self.optimizer = optimizer

        self.loss_recorder = LossRecorder()


    ## setter functions for the class variables
    def set_network(self, arch_build_function):
        self.network = arch_build_function(self.input_res)
        self.arch_build_function = arch_build_function

    def set_resolutions(self, resolutions):
        self.input_res, self.output_res = resolutions

    def set_weights(self, weights):
        self.network.set_weights(weights)

    def set_loss_function(self, loss_build_function):
        self.loss_function = loss_build_function(input_res=self.output_res)
        self.loss_build_function = loss_build_function

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_loss_recorder(self, loss_recorder):
        self.loss_recorder = loss_recorder

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = True

        # Arch Build Function
        if self.arch_build_function is None:
            print(TColors.NOTE + 'Arch Build Function: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Arch Build Function: ' + TColors.OKGREEN + 'available')
        # Network
        if self.network is None:
            print(TColors.NOTE + 'Network: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Network: ' + TColors.OKGREEN + 'available')
         # Loss Build Function
        if self.loss_build_function is None:
            print(TColors.NOTE + 'Loss Build Function: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Loss Build Function: ' + TColors.OKGREEN + 'available')
        # Loss Function
        if self.loss_function is None:
            print(TColors.NOTE + 'Loss Function: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Loss Function: ' + TColors.OKGREEN + 'available')
        # Resolutions
        if self.input_res is None:
            print(TColors.NOTE + 'Input Resolution: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Input Resolution: ' + TColors.OKGREEN + 'available')
        if self.output_res is None:
            print(TColors.NOTE + 'Output Resolution: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Output Resolution: ' + TColors.OKGREEN + 'available')
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
            'arch_build_function': self.arch_build_function,
            'loss_build_function': self.loss_build_function,
            'resolutions': (self.input_res, self.output_res),
            'optimizer': self.optimizer,
            'loss_recorder': self.loss_recorder
        }

    # this function loads all the given values into class variables
    def load_variables(self, variables):
        self.set_resolutions(variables['resolutions'])

        self.set_network(variables['arch_build_function'])
        self.network.set_weights(variables['weights'])

        self.set_loss_function(variables['loss_build_function'])
        self.set_optimizer(variables['optimizer'])
        self.set_loss_recorder(variables['loss_recorder'])