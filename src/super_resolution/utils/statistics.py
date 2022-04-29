### statistics.py ###
# This module contains the StatsRecorder class. This class
# is used by a Model class and records the loss.
##

from telnetlib import SE
import numpy as np

## Plot Params ##
PRIMARY_LW = 1
SECONDARY_LW = 0.7
AX_LABEL_FS = 10
LABEL_FS = 7

# Stats recorder class records the loss and a chosen metric
class StatsRecorder():
    def __init__(self):
        self.loss = []

        self.feature_load_time = []
        self.label_load_time = []
        self.train_time = []

        self.cpu_load = []
        self.ram_load = []
        self.gpu_load = []

    # add a new loss value to loss
    def add_loss(self, loss):
        if loss is not None:
            # convert to scalar
            scalar_loss = loss.numpy()
            self.loss.append(scalar_loss)

    # add a new time to times
    def add_time(self, time):
        if time is not None:
            # unpack
            f_time, l_time, t_time = time

            if f_time is not None:
                self.feature_load_time.append(f_time)
            if l_time is not None:
                self.label_load_time.append(l_time)
            if t_time is not None:
                self.train_time.append(t_time)

    # add a new load value to loads
    def add_sys_load(self, sys_load):
        if sys_load is not None:
            # unpack
            cpu_load, ram_load, gpu_load = sys_load

            if cpu_load is not None:
                self.cpu_load.append(cpu_load)
            if ram_load is not None:
                self.ram_load.append(ram_load)
            if gpu_load is not None:
                self.gpu_load.append(gpu_load)

    # add a new metric value to loads
    def add_metrics(self, metrics):
        if metrics is not None:
            pass

    def plot_loss(self, ax, epochs):
        # get y - axis
        y = self.loss
        y_avg = []

        step = int(np.ceil(len(y)/epochs))
        for epoch in range(epochs):
            y_avg.append(np.mean(y[epoch*step:(epoch+1)*step]))

        # make x - axis
        x = np.linspace(0, epochs, num=len(y))
        x_avg = np.linspace(0, epochs, num=epochs)
        
        # set labels
        ax.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax.set_ylabel('Loss', fontsize=AX_LABEL_FS)
        ax.label_outer()

        # plot
        ax.plot(x, y, linewidth=SECONDARY_LW, linestyle=':', color='gray', label='each batch')
        ax.plot(x_avg, y_avg, linewidth=PRIMARY_LW, color='red', label='average over epoch')
        ax.legend(loc='upper right', fontsize=LABEL_FS)

    def plot_time(self, ax, epochs):
        # get y - axis
        feature_avg = np.mean(self.feature_load_time)
        label_avg = np.mean(self.label_load_time)
        train_avg = np.mean(self.train_time)

        # make x - axis
        x = np.linspace(0, epochs, num=len(self.feature_load_time))
        
        # set labels
        ax.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax.set_ylabel('Time per Batch (s)', fontsize=AX_LABEL_FS)
        ax.label_outer()

        # plot
        ax.plot(x, self.feature_load_time, linewidth=PRIMARY_LW, color='tab:blue', label='feature load time')
        ax.plot(x, self.label_load_time, linewidth=PRIMARY_LW, color='tab:green', label='label load time')
        ax.plot(x, self.train_time, linewidth=PRIMARY_LW, color='tab:orange', label='train time')

        ax.legend(loc='upper right', fontsize=LABEL_FS)

    def plot_time(self, ax, epochs):
        # make x - axis
        x = np.linspace(0, epochs, num=len(self.feature_load_time))
        
        # set labels
        ax.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax.set_ylabel('Time per Batch (s)', fontsize=AX_LABEL_FS)
        ax.label_outer()

        # plot
        ax.plot(x, self.feature_load_time, linewidth=PRIMARY_LW, color='tab:blue', label='feature load time')
        ax.plot(x, self.label_load_time, linewidth=PRIMARY_LW, color='tab:green', label='label load time')
        ax.plot(x, self.train_time, linewidth=PRIMARY_LW, color='tab:orange', label='train time')

        ax.legend(loc='upper right', fontsize=LABEL_FS)

    def plot_sys_load(self, ax, epochs):
        # make x - axis
        x = np.linspace(0, epochs, num=len(self.cpu_load))
        
        # set labels
        ax.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax.set_ylabel('Load (%)', fontsize=AX_LABEL_FS)
        ax.label_outer()

        # plot
        ax.plot(x, self.cpu_load, linewidth=PRIMARY_LW, color='tab:orange', label='cpu load')
        ax.plot(x, self.ram_load, linewidth=PRIMARY_LW, color='tab:blue', label='ram load')
        ax.plot(x, self.gpu_load, linewidth=PRIMARY_LW, color='tab:green', label='gpu load')

        ax.legend(loc='upper right', fontsize=LABEL_FS)
