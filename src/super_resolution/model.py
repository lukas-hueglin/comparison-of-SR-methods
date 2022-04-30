### model.py ###
# This module defines the Model class. This class is a container for:
#   - a network (built with a make_ function)
#   - a loss function (defined with a _loss function)
#   - a tensorflow optimizer
##

from utils import StatsRecorder

class Model():
    def __init__(self, network = None, loss_function = None, optimizer = None, metric_functions=[]):
        self.network, self.network_name = network
        self.loss_function = loss_function
        self.optimizer = optimizer

        self.stats_recorder = StatsRecorder(metric_functions)


    ## setter functions for the class variables
    def set_network(self, network):
        self.network, self.network_name = network

    def set_loss_function(self, loss_function):
        self.loss_function = loss_function

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_metric(self, metric):
        self.stats_recoder.metric_function = metric

    # This function is used to create the ABOUT.md file it returns the network name
    def get_info(self):
        # get the name of the loss function
        lf_name = self.loss_function.__name__
        lf_name = lf_name.replace('_loss', '')

        # get the learning rate
        lr = self.optimizer.lr.numpy()

        return self.network_name, lf_name, lr