### statistics.py ###
# This module contains the StatsRecorder class. This class
# is used by a Model class and records the loss.
##

import numpy as np
import cv2

## metrics
import tensorflow as tf
def PSNR_metric(y_pred, y_true):
    sum = 0
    for i in range(len(y_pred)):
        img_y_pred = np.reshape(y_pred[i], -1)
        img_y_true = np.reshape(y_true[i], -1)

        mse = np.sum(np.square(img_y_true - img_y_pred))/len(img_y_pred)
        sum += 10 * np.log10(1/mse)
    return sum/len(y_pred)

def SSIM_metric(y_pred, y_true):
    k1 = 0.01
    k2 = 0.03
    L = 1.0 # Dynamic range

    C1 = np.square(k1*L)
    C2 = np.square(k2*L)
    C3 = C2/2

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
SECONDARY_LW = 0.4
AX_LABEL_FS = 10
LABEL_FS = 7

# Loss recorder class records the loss of a model
class LossRecorder():
    def __init__(self):
        self.loss = []
        self.epochs = 1

    # add a new loss value to loss
    def add_loss(self, loss):
        if loss is not None:
            # convert to scalar
            scalar_loss = loss.numpy()
            self.loss.append(scalar_loss)

    # count the epoch counter up one epoch.
    def add_epoch(self):
        self.epochs += 1

    # fills the given subplots with the loss data
    def plot_loss(self, ax):
        # get y - axis
        y = self.loss
        y_avg = []

        step = int(np.ceil(len(y)/self.epochs))
        for epoch in range(self.epochs):
            y_avg.append(np.mean(y[epoch*step:(epoch+1)*step]))

        # make x - axis
        x = np.linspace(0, self.epochs, num=len(y))
        x_avg = np.linspace(0, self.epochs, num=self.epochs)
        
        # set labels
        ax.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax.set_ylabel('Loss', fontsize=AX_LABEL_FS)
        ax.label_outer()

        # plot
        ax.plot(x, y, linewidth=SECONDARY_LW, linestyle=':', color='gray', label='each batch')
        ax.plot(x_avg, y_avg, linewidth=PRIMARY_LW, color='red', label='average over epoch')
        ax.legend(loc='best', fontsize=LABEL_FS)

# Stats recorder class records times, system loads and metrics
class StatsRecorder():
    def __init__(self, metric_functions):
        self.training = {
            'feature_load_time': [],
            'label_load_time': [],
            'network_time': [],
            'total_time': 0,
            'cpu_load': [],
            'ram_load': [],
            'gpu_load': [],
            'metrics': [[] for _ in range(len(metric_functions))]
        }

        self.validation = {
            'feature_load_time': [],
            'label_load_time': [],
            'network_time': [],
            'total_time': 0,
            'cpu_load': [],
            'ram_load': [],
            'gpu_load': [],
            'metrics': [[] for _ in range(len(metric_functions))]
        }

        self.metric_functions = metric_functions
        self.epochs = 1

    # add a new time
    def add_time(self, time, train=True):
        data_pack = self.training if train else self.validation

        if time is not None:
            # unpack
            f_time, l_time, t_time, total_time = time

            if f_time is not None:
                data_pack['feature_load_time'].append(f_time)
            if l_time is not None:
                data_pack['label_load_time'].append(l_time)
            if t_time is not None:
                data_pack['network_time'].append(t_time)
            if total_time is not None:
                data_pack['total_time'] += total_time

    # add a new load value
    def add_sys_load(self, sys_load, train=True):
        data_pack = self.training if train else self.validation

        if sys_load is not None:
            # unpack
            cpu_load, ram_load, gpu_load = sys_load

            if cpu_load is not None:
                data_pack['cpu_load'].append(cpu_load)
            if ram_load is not None:
                data_pack['ram_load'].append(ram_load)
            if gpu_load is not None:
                data_pack['gpu_load'].append(gpu_load)

    # add a new metric value
    def add_metrics(self, metrics, train=True):
        data_pack = self.training if train else self.validation

        if metrics is not None:
            y_pred, y_true = metrics
            for i in range(len(data_pack['metrics'])):
                data_pack['metrics'][i].append(self.metric_functions[i]((y_pred+1)/2, (y_true+1)/2))

    # count the epoch counter up one epoch.
    def add_epoch(self):
        self.epochs += 1

    # fills the given subplots with the time data
    def plot_time(self, ax_train, ax_validation, train=True):
        ## training data
        # make x - axis
        x_train = np.linspace(0, self.epochs, num=len(self.training['feature_load_time']))
        
        # set title
        ax_train.title.set_text('Training')

        # set axis scale
        ax_train.set_yscale('log')

        # set labels
        ax_train.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax_train.set_ylabel('Time (s)', fontsize=AX_LABEL_FS)

        # plot
        ax_train.plot(x_train, self.training['feature_load_time'], linewidth=PRIMARY_LW, color='tab:blue', label='feature load time')
        ax_train.plot(x_train, self.training['label_load_time'], linewidth=PRIMARY_LW, color='tab:green', label='label load time')
        ax_train.plot(x_train, self.training['network_time'], linewidth=PRIMARY_LW, color='tab:orange', label='train time')

        ax_train.legend(loc='best', fontsize=LABEL_FS)

        ## validation data
        if not train:
            # make x - axis
            x_validation = np.linspace(0, 1, num=len(self.validation['feature_load_time']))
            
            # set title
            ax_validation.title.set_text('Validation')

            # set axis scale
            ax_validation.set_yscale('log')

            # set labels
            ax_validation.set_xlabel('Validation Process', fontsize=AX_LABEL_FS)
            ax_validation.set_ylabel('Time (s)', fontsize=AX_LABEL_FS)

            # plot
            ax_validation.plot(x_validation, self.validation['feature_load_time'], linewidth=PRIMARY_LW, color='tab:blue', label='feature load time')
            ax_validation.plot(x_validation, self.validation['label_load_time'], linewidth=PRIMARY_LW, color='tab:green', label='label load time')
            ax_validation.plot(x_validation, self.validation['network_time'], linewidth=PRIMARY_LW, color='tab:orange', label='generation time')

            ax_validation.legend(loc='best', fontsize=LABEL_FS)

    # fills the given subplots with the system load data
    def plot_sys_load(self, fig_train, fig_validation, train=True):
        # unwrap axes
        ax_t_cpu, ax_t_ram, ax_t_gpu = fig_train.subplots(3)

        ## training data
        # make x - axis
        x_train = np.linspace(0, self.epochs, num=len(self.training['cpu_load']))

        # make average
        x_t_cpu_avg = np.mean(self.training['cpu_load'])
        x_t_ram_avg = np.mean(self.training['ram_load'])
        x_t_gpu_avg = np.mean(self.training['gpu_load'])

        # set title
        fig_train.suptitle('Training')
        
        # set labels
        ax_t_cpu.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax_t_cpu.set_ylabel('Load (%)', fontsize=AX_LABEL_FS)
        ax_t_cpu.label_outer()
        ax_t_ram.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax_t_ram.set_ylabel('Load (%)', fontsize=AX_LABEL_FS)
        ax_t_ram.label_outer()
        ax_t_gpu.set_xlabel('Epochs', fontsize=AX_LABEL_FS)
        ax_t_gpu.set_ylabel('Load (%)', fontsize=AX_LABEL_FS)
        ax_t_gpu.label_outer()

        # set limits
        ax_t_cpu.set_ylim([0, 100])
        ax_t_ram.set_ylim([0, 100])
        ax_t_gpu.set_ylim([0, 100])

        # plot
        ax_t_cpu.plot(x_train, self.training['cpu_load'], linewidth=SECONDARY_LW, linestyle=':', color='sandybrown')
        ax_t_cpu.hlines(x_t_cpu_avg, 0, self.epochs, linewidth=PRIMARY_LW, color='tab:orange', label='cpu load')
        ax_t_ram.plot(x_train, self.training['ram_load'], linewidth=SECONDARY_LW, linestyle=':', color='cornflowerblue')
        ax_t_ram.hlines(x_t_ram_avg, 0, self.epochs, linewidth=PRIMARY_LW, color='tab:blue', label='ram load')
        ax_t_gpu.plot(x_train, self.training['gpu_load'], linewidth=SECONDARY_LW, linestyle=':', color='palegreen')
        ax_t_gpu.hlines(x_t_gpu_avg, 0, self.epochs, linewidth=PRIMARY_LW, color='tab:green', label='gpu load')

        # set legend
        if train:
            ax_t_cpu.legend(loc='upper right', fontsize=LABEL_FS)
            ax_t_ram.legend(loc='upper right', fontsize=LABEL_FS)
            ax_t_gpu.legend(loc='upper right', fontsize=LABEL_FS)

        ## validation data
        if not train:
            # unwrap axes
            ax_v_cpu, ax_v_ram, ax_v_gpu = fig_validation.subplots(3)

            # make x - axis
            x_validation = np.linspace(0, 1, num=len(self.validation['cpu_load']))

             # make average
            x_v_cpu_avg = np.mean(self.validation['cpu_load'])
            x_v_ram_avg = np.mean(self.validation['ram_load'])
            x_v_gpu_avg = np.mean(self.validation['gpu_load'])

            # set title
            fig_validation.suptitle('Validation')
            
            # set labels
            ax_v_cpu.set_xlabel('Validation Process', fontsize=AX_LABEL_FS)
            ax_v_cpu.label_outer()
            ax_v_ram.set_xlabel('Validation Process', fontsize=AX_LABEL_FS)
            ax_v_ram.label_outer()
            ax_v_gpu.set_xlabel('Validation Process', fontsize=AX_LABEL_FS)
            ax_v_gpu.label_outer()

            # set limits
            ax_v_cpu.set_ylim([0, 100])
            ax_v_ram.set_ylim([0, 100])
            ax_v_gpu.set_ylim([0, 100])

            # plot
            ax_v_cpu.plot(x_validation, self.validation['cpu_load'], linewidth=SECONDARY_LW, linestyle=':', color='sandybrown')
            ax_v_cpu.hlines(x_v_cpu_avg, 0, 1, linewidth=PRIMARY_LW, color='tab:orange', label='cpu load')
            ax_v_ram.plot(x_validation, self.validation['ram_load'], linewidth=SECONDARY_LW, linestyle=':', color='cornflowerblue')
            ax_v_ram.hlines(x_v_ram_avg, 0, 1, linewidth=PRIMARY_LW, color='tab:blue', label='ram load')
            ax_v_gpu.plot(x_validation, self.validation['gpu_load'], linewidth=SECONDARY_LW, linestyle=':', color='palegreen')
            ax_v_gpu.hlines(x_v_gpu_avg, 0, 1, linewidth=PRIMARY_LW, color='tab:green', label='gpu load')

            # set legend
            ax_v_cpu.legend(loc='upper right', fontsize=LABEL_FS)
            ax_v_ram.legend(loc='upper right', fontsize=LABEL_FS)
            ax_v_gpu.legend(loc='upper right', fontsize=LABEL_FS)

    # fills the given subplots with the metric data
    def plot_metrics(self, axs, train=True):
        for i in range(len(self.metric_functions)):
            ## training data
            # get y - axis
            y_train = self.training['metrics'][i]
            y_train_avg = []

            step = int(np.ceil(len(y_train)/self.epochs))
            for epoch in range(self.epochs):
                y_train_avg.append(np.mean(y_train[epoch*step:(epoch+1)*step]))

            # make x - axis
            x_train = np.linspace(0, self.epochs, num=len(y_train))
            x_train_avg = np.linspace(0, self.epochs, num=self.epochs)

            # set title
            axs[i].title.set_text((self.metric_functions[i].__name__).split('_')[0])
            
            # set labels
            axs[i].set_xlabel('Epochs', fontsize=AX_LABEL_FS)
            axs[i].set_ylabel('Accuracy', fontsize=AX_LABEL_FS)
            axs[i].label_outer()

            # plot
            axs[i].plot(x_train, y_train, linewidth=SECONDARY_LW, linestyle=':', color='gray', label='each batch')
            axs[i].plot(x_train_avg, y_train_avg, linewidth=PRIMARY_LW, color='red', label='average over epoch')
            axs[0].legend(loc='best', fontsize=LABEL_FS)

            ## validation data
            if not train:
                # get average
                y_validation_avg = np.mean(self.validation['metrics'][i])

                #plot
                axs[i].hlines(y_validation_avg, 0, self.epochs, linewidth=PRIMARY_LW, color='darkred', label='validation average')

