### statistics.py ###
# This module contains the StatsRecorder class. This class
# is used by a Model class and records the loss.
##

import numpy as np
import cv2

## metrics

def PSNR_metric(y_pred, y_true):
    sum = 0
    for i in range(len(y_pred)):
        img_y_pred = np.reshape(y_pred[i], -1)
        img_y_true = np.reshape(y_true[i], -1)

        mse = np.sum(np.square(img_y_true - img_y_pred))/len(img_y_pred)
        sum += 10 * np.log10(1/mse)
    return sum/len(y_pred)

def SSIM_metric(y_pred, y_true):
    k1 = 0.5
    k2 = 0.5
    L = 1.0 # Dynamic range

    C1 = np.square(k1*L)
    C2 = np.square(k2*L)
    C3 = 1.0

    alpha = 1.0
    beta = 1.0
    gamma = 1.0

    sum = 0
    for i in range(len(y_pred)):
        gray_y_pred = cv2.cvtColor(np.array(y_pred[i]*255, dtype="uint8"), cv2.COLOR_RGB2GRAY)/255
        gray_y_true = cv2.cvtColor(np.array(y_true[i]*255, dtype="uint8"), cv2.COLOR_RGB2GRAY)/255
        img_y_pred = np.reshape(gray_y_pred, -1)
        img_y_true = np.reshape(gray_y_true, -1)

        luminance_pred = np.mean(img_y_pred)
        luminance_true = np.mean(img_y_true)

        contrast_pred = np.sqrt(np.sum(np.square(img_y_pred-luminance_pred))/(len(img_y_pred)-1))
        contrast_true = np.sqrt(np.sum(np.square(img_y_true-luminance_true))/(len(img_y_true)-1))

        convariance = np.sum((img_y_pred-luminance_pred) * (img_y_true-luminance_true))/(len(img_y_pred)-1)

        comp_luminance = (2*luminance_pred*luminance_true + C1)/(np.square(luminance_pred) + np.square(luminance_true) + C1)
        comp_contrast = (2*contrast_pred*contrast_true + C2)/(np.square(contrast_pred) + np.square(contrast_true) + C2)
        comp_stability = (convariance + C3)/(contrast_pred*contrast_true + C3)

        sum += np.power(comp_luminance, alpha) * np.power(comp_contrast, beta) * np.power(comp_stability, gamma)
    return sum/len(y_pred)


## Plot Params ##
PRIMARY_LW = 1
SECONDARY_LW = 0.7
AX_LABEL_FS = 10
LABEL_FS = 7

# Stats recorder class records the loss and a chosen metric
class StatsRecorder():
    def __init__(self, metric_functions):
        self.loss = []

        self.feature_load_time = []
        self.label_load_time = []
        self.train_time = []

        self.cpu_load = []
        self.ram_load = []
        self.gpu_load = []

        self.metric_functions = metric_functions
        self.metrics = [[] for _ in range(len(metric_functions))]

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
            y_pred, y_true = metrics
            for i in range(len(self.metrics)):
                self.metrics[i].append(self.metric_functions[i](y_pred, y_true))


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
        ax.legend(loc='best', fontsize=LABEL_FS)

    def plot_time(self, ax, epochs):
        # make x - axis
        x = np.linspace(0, epochs, num=len(self.feature_load_time))
        
        # set labels
        ax.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax.label_outer()

        # plot
        ax.plot(x, self.feature_load_time, linewidth=PRIMARY_LW, color='tab:blue', label='feature load time')
        ax.plot(x, self.label_load_time, linewidth=PRIMARY_LW, color='tab:green', label='label load time')
        ax.plot(x, self.train_time, linewidth=PRIMARY_LW, color='tab:orange', label='train time')

        ax.legend(loc='best', fontsize=LABEL_FS)

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

        ax.legend(loc='best', fontsize=LABEL_FS)

    def plot_metrics(self, axs, epochs):
        for i in range(len(self.metric_functions)):
            # get y - axis
            y = self.metrics[i]
            y_avg = []

            step = int(np.ceil(len(y)/epochs))
            for epoch in range(epochs):
                y_avg.append(np.mean(y[epoch*step:(epoch+1)*step]))

            # make x - axis
            x = np.linspace(0, epochs, num=len(y))
            x_avg = np.linspace(0, epochs, num=epochs)

            # set title
            axs[i].title.set_text((self.metric_functions[i].__name__).split('_')[0])
            
            # set labels
            axs[i].set_xlabel('Epochs', fontsize=AX_LABEL_FS)
            axs[i].set_ylabel('Accuracy', fontsize=AX_LABEL_FS)
            axs[i].label_outer()

            # plot
            axs[i].plot(x, y, linewidth=SECONDARY_LW, linestyle=':', color='gray', label='each batch')
            axs[i].plot(x_avg, y_avg, linewidth=PRIMARY_LW, color='red', label='average over epoch')
            axs[0].legend(loc='best', fontsize=LABEL_FS)

