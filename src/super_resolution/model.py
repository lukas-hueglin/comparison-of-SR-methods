### model.py ###

class Model():
    def __init__(self, network = None, loss_function = None, optimizer = None):
        self.network = network
        self.loss_function = loss_function
        self.optimizer = optimizer

    def set_network(self, network):
        self.network = network

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer