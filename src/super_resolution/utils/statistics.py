### statistics.py ###
# This module contains the StatsRecorder class. This class
# is used by a Model class and records the loss.
##

# Stats recorder class records the loss and a chosen metric
class StatsRecorder():
    def __init__(self):
        self.tracked_loss = []

    # add a new loss value to tracked_loss
    def add_loss(self, loss):
        # convert to scalar
        scalar_loss = loss.numpy()
        self.tracked_loss.append(scalar_loss)

