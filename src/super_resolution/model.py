### model.py ###
# This module defines the Model class. This class is a container for:
#   - a network (built with a make_ function)
#   - a loss function (defined with a _loss function)
#   - a tensorflow optimizer
##

class Model():
    def __init__(self, network = None, loss_function = None, optimizer = None):
        self.network = network
        self.loss_function = loss_function
        self.optimizer = optimizer


    ## setter functions for the class variables
    def set_network(self, network):
        self.network = network

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer