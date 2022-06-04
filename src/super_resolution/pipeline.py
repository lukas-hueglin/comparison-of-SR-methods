### pipeline.py ###
# This module connects the framework with the data. It
# is a container for all it needs to train or test out a framework.
##

import tensorflow as tf

import os
import datetime
import time

import dill

import cv2
import imageio

import psutil
import GPUtil

import numpy as np
from tqdm import tqdm

from abc import ABC, abstractmethod

from utils import TColors


## helper_functions ##

# Function which creates a animated gif from all progress images
def create_gif(path):
    # get all directories (all different images)
    dirs = os.listdir(path)

    for d in dirs:
        # make new path
        dir_path = os.path.join(path, d)

        # get all images (epochs)
        images = os.listdir(dir_path)

        # make file name
        file_name = os.path.join(dir_path, 'animfile.gif')

        with imageio.get_writer(file_name, mode='I') as writer:
            # iterate all images
            for i in images:
                if i.endswith(".jpg"):
                    image_path = os.path.join(dir_path, i)
                    image = imageio.imread(image_path)
                    writer.append_data(image)

# returns the filepath with the next version index
def get_next_version(raw_path):
    index = 2
    output_path = raw_path + '_v.' + f"{1:03d}"

    # check if the path is already used and count up
    while os.path.exists(output_path):
        output_path = raw_path + '_v.' f"{index:03d}"
        index += 1

    return output_path


## Pipeline class ##
class Pipeline(ABC):
    def __init__(self, framework=None):
        self.framework = framework

        self.model_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, 'models'))
        if framework is not None:
            self.output_path = get_next_version(os.path.join(self.model_path, self.framework.name))

    ## setter functions for the class variables.
    def set_framework(self, framework):
        self.framework = framework
        self.output_path = get_next_version(os.path.join(self.model_path, self.framework.name))

    # this function loads all the given values into class
    # variables and writes them to a byte file with dill
    def load_framework(self, framework_path):
        self.output_path = os.path.join(self.model_path, framework_path)

        dir_path = os.path.join(self.output_path,'checkpoints')
        file_path = os.path.join(dir_path, os.listdir(dir_path)[-1])

        with open(file_path, 'rb') as file:
            variables = dill.load(file)

        self.framework = variables['class']()
        self.framework.load_variables(variables)

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    @ abstractmethod
    def check_variables(self):
        # Framework
        if self.framework is None:
            print(TColors.NOTE + 'Framework: ' + TColors.FAIL + 'not available')
            return False
        else:
            print(TColors.NOTE + 'Framework: ' + TColors.OKGREEN + 'available')
            return True

    # empty function for checking all the variables of the pipeline and the framework
    # This functions should only be called as the first functions using the network
    @abstractmethod
    def check(self):
        pass

    # checks all the variables of the framework
    def check_framework(self):
        # print 'Check Framework'
        print(TColors.HEADER + 'Check Framework:\n' + TColors.ENDC)

        # generate noise
        res = self.framework.input_res
        noise = tf.random.normal([1, res, res, 3])

        # generate image
        self.framework.generate_images(noise, check=True)

        # clear the console font
        print(TColors.ENDC)


class Trainer(Pipeline):
    def __init__(self, framework=None, dataset_loader=None, sample_loader=None, epochs=1):
        super().__init__(framework)

        self.dataset_loader = dataset_loader

        self.sample_loader = sample_loader
        if self.sample_loader is not None:
            self.sample_images = self.sample_loader.load_samples()
        else:
            self.sample_images = None

        self.epochs = epochs

    ## setter functions for the class variables.
    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_dataset_loader(self, dataset_loader):
        self.dataset_loader = dataset_loader

    def set_sample_loader(self, sample_loader):
        self.sample_loader = sample_loader
        self.sample_images = self.sample_loader.load_samples()


    # The main train function. This function gets called by the user.
    def train(self):
        # check if everything is specified
        if self.dataset_loader != None:

            # iterate over each epoch
            start_epoch = self.framework.stats_recorder.epochs
            for epoch in range(start_epoch, self.epochs + start_epoch):  
                # print a message
                print(TColors.HEADER + '\nEpoch ' + str(epoch) + ':\n' +TColors.ENDC)

                # prepare the data loading process
                self.dataset_loader.prepare_loading(train=True)

                # iterate over each batch
                num_batches = int(np.ceil(self.dataset_loader.train_size/self.dataset_loader.batch_size))
                for batch in tqdm(range(num_batches)):
                    # get data to train
                    (features, feature_time), (labels, label_time) = self.dataset_loader.access_loading()

                    # convert to tensors
                    features = tf.convert_to_tensor(features)
                    labels = tf.convert_to_tensor(labels)

                    # train
                    now = time.perf_counter()
                    generated_images, loss = self.framework.train_step(features, labels)

                    # stop timer
                    network_time = time.perf_counter() - now

                    # generate sys_load
                    cpu_load = psutil.cpu_percent(interval=None, percpu=False)
                    ram_load = psutil.virtual_memory().percent
                    gpu_load_list = []
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_load_list.append(gpu.load * 100)
                    gpu_load = np.mean(gpu_load_list)

                    # add values to stats recorder
                    self.framework.update_stats_recorder(
                        loss=loss,
                        sys_load=(cpu_load, ram_load, gpu_load),
                        time=(feature_time, label_time, network_time),
                        metrics=(generated_images, labels),
                        train=True
                    )

                # join the processes
                self.dataset_loader.close_loading()

                # Generate sample image
                if self.output_path != None and self.sample_images != None:
                    self.perform_SR(epoch)

                # add a epoch to the stats recorders
                if epoch != self.epochs + start_epoch - 1:
                    self.framework.add_epoch()

                # save checkpoint
                if epoch % 1 == 0 or epoch == self.epochs + start_epoch - 1: # normal epoch % 10 == 0
                    self.save_framework(epoch)

            # plot stats
            plot_path = os.path.join(self.output_path, 'statistics')
            name = self.framework.name
            self.framework.plot_and_save_stats(plot_path, name, train=True)

            # create gif
            image_path = image_path = os.path.join(self.output_path, 'progress_images')
            create_gif(image_path)

            # make ABOUT.md file
            self.create_ABOUT_file()

        else:
            print(TColors.WARNING + 'The training dataset or the framework is not specified!' + TColors.ENDC)

    # Checks the variables of the pipeline and the framework
    # This functions should only be called as the first functions using the network
    def check(self):
        # print 'Check Trainer'
        print(TColors.HEADER + '\nCheck Trainer:\n' + TColors.ENDC)

        assert(self.check_variables())

        # check framework
        super().check_framework()

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = super().check_variables()

        # Dataset Loader
        if self.dataset_loader is None:
            print(TColors.NOTE + 'Dataset Loader: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Dataset Loader: ' + TColors.OKGREEN + 'available')
        # Sample Loader
        if self.sample_loader is None:
            print(TColors.NOTE + 'Sample Loader: ' + TColors.WARNING + 'not available')
            # it's okay if it isn't specified
        else:
            print(TColors.NOTE + 'Sample Loader: ' + TColors.OKGREEN + 'available')
        # Epochs
        if self.epochs == 1:
            print(TColors.NOTE + 'Epochs: ' + TColors.WARNING + 'is 1')
            # it's okay if it isn't specified
        else:
            print(TColors.NOTE + 'Epochs: ' + TColors.OKGREEN + 'is ' + str(self.epochs))

        # make python print normal
        print(TColors.ENDC)

        return status_ok

    # performes the superresolution task and returns the upsampled image
    def perform_SR(self, epoch):
        # generate images
        generated_images = self.framework.generate_images(self.sample_images)
        self.save_sample_images(generated_images, epoch)

    # performes superresolution on the sample images
    def save_sample_images(self, images, epoch):
        for i in range(len(images)):
            img = cv2.cvtColor(np.array(images[i])*255, cv2.COLOR_RGB2BGR)

            # make paths
            dir_path = os.path.join(self.output_path, 'progress_images', 'image_' + f"{i+1:02d}")
            img_path = os.path.join(dir_path, 'epoch_' + f"{epoch:03d}" + '.jpg')

            # make a new folder
            os.makedirs(dir_path, exist_ok=True)
            
            cv2.imwrite(img_path, img)

    # This function is used to create the ABOUT.md file
    def create_ABOUT_file(self):
        # get info from framework
        text = self.framework.get_info()
        # add 3 - Training parameters
        text += '## 3 - Training parameters\n\n'
        text += 'Date: ' + str(datetime.date.today()) + '\n\n'
        text += 'Epochs: ' + str(self.framework.stats_recorder.epochs) + '</br>\n'
        text += 'Batch size: ' + str(self.dataset_loader.batch_size) + '</br>\n'
        text += 'Buffer size: ' + str(self.dataset_loader.buffer_size) + '\n\n'
        # add 4 - datasets (in future)
        text += '## 4 - Datasets\n\n'
        text += 'Dataset: ' + self.dataset_loader.path + ' </br>\n'
        text += 'Dataset size: ' + str(self.dataset_loader.dataset_size) + ' </br>\n'
        text += 'Training - Validation ratio: ' + str(self.dataset_loader.train_ratio) + '\n\n'
        # add 5 - Sample images
        text += '## 5 - Sample Images\n\n'
        text += '*Note: All these images are available under [progress images](./progress_images/)*\n\n'

        image_path = os.path.join(self.output_path, 'progress_images')
        image_dirs = os.listdir(image_path)
        for image in image_dirs:
            text += '![' + image + '](./progress_images/' + image + '/animfile.gif)\n'

        # add 6 - Notes
        text += '\n## 6 - Notes\n\n'
        text += '*Placeholder*'

        # write file
        file = open(self.output_path+'\\ABOUT.md', 'w')
        file.write(text)
        file.close()

    # this function saves all class variables into a directory
    def save_framework(self, epoch):
        variables = self.framework.save_variables()

        ckpt_path = os.path.join(self.output_path, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)

        file_path = os.path.join(ckpt_path, 'epoch_' + f"{epoch:02d}" + '.ckpt')
        with open(file_path, 'wb') as file:
            dill.dump(variables, file)
    
class Validator(Pipeline):
    def __init__(self, framework=None, dataset_loader=None):
        super().__init__(framework)
        self.dataset_loader = dataset_loader

    ## setter functions for the class variables.
    def set_dataset_loader(self, dataset_loader):
        self.dataset_loader = dataset_loader

    # validates the neural network using the validation dataset
    def validate(self):
        # print a message
        print(TColors.HEADER + '\nValidation:\n' +TColors.ENDC)

        # prepare the data loading process
        self.dataset_loader.prepare_loading(train=False)

        # iterate over each batch
        num_batches = int(np.ceil(self.dataset_loader.validation_size/self.dataset_loader.batch_size))
        for batch in tqdm(range(num_batches)):
            # get data to train
            (features, feature_time), (labels, label_time) = self.dataset_loader.access_loading()

            # convert to tensors
            features = tf.convert_to_tensor(features)
            labels = tf.convert_to_tensor(labels)

            # train
            now = time.perf_counter()
            generated_images, loss = self.framework.train_step(features, labels, train=False)

            # stop timer
            network_time = time.perf_counter() - now

            # generate sys_load
            cpu_load = psutil.cpu_percent(interval=None, percpu=False)
            ram_load = psutil.virtual_memory().percent
            gpu_load_list = []
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_load_list.append(gpu.load * 100)
            gpu_load = np.mean(gpu_load_list)

            # add values to stats recorder
            self.framework.update_stats_recorder(
                loss=loss,
                sys_load=(cpu_load, ram_load, gpu_load),
                time=(feature_time, label_time, network_time),
                metrics=(generated_images, labels),
                train=False
            )

        # join the processes
        self.dataset_loader.close_loading()

        # plot stats
        plot_path = os.path.join(self.output_path, 'statistics')
        name = self.framework.name
        self.framework.plot_and_save_stats(plot_path, name, train=False)

    # Checks the variables of the pipeline and the framework
    # This functions should only be called as the first functions using the network
    def check(self):
        # print 'Check Trainer'
        print(TColors.HEADER + '\nCheck Trainer:\n' + TColors.ENDC)

        assert(self.check_variables())

        # check framework
        super().check_framework()

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = super().check_variables()

        # Dataset Loader
        if self.dataset_loader is None:
            print(TColors.NOTE + 'Dataset Loader: ' + TColors.FAIL + 'not available')
            status_ok = False
        else:
            print(TColors.NOTE + 'Dataset Loader: ' + TColors.OKGREEN + 'available')

        # make python print normal
        print(TColors.ENDC)

        return status_ok


class Performer(Pipeline):
    def __init__(self, framework=None, sample_loader=None):
        super().__init__(framework)

        self.sample_loader = sample_loader
        if self.sample_loader is not None:
            self.sample_images = self.sample_loader.load_samples()
        else:
            self.sample_images = None

     ## setter functions for the class variables.
    def set_sample_loader(self, sample_loader):
        self.sample_loader = sample_loader
        self.sample_images = self.sample_loader.load_samples()

    def set_output_path(self, output_path):
        self.output_path = output_path

    # Checks the variables of the pipeline and the framework
    # This functions should only be called as the first functions using the network
    def check(self):
        # print 'Check Trainer'
        print(TColors.HEADER + '\nCheck Trainer:\n' + TColors.ENDC)

        assert(self.check_variables())

        # check framework
        super().check_framework()

    # checks if all variables are specified and if the program can be ran
    # it also prints all the results to the console 
    def check_variables(self):
        status_ok = super().check_variables()

        # Sample Loader
        if self.sample_loader is None:
            print(TColors.NOTE + 'Sample Loader: ' + TColors.WARNING + 'not available')
            # it's okay if it isn't specified
        else:
            print(TColors.NOTE + 'Sample Loader: ' + TColors.OKGREEN + 'available')

        # make python print normal
        print(TColors.ENDC)

        return status_ok

    # performes the superresolution task and returns the upsampled image
    def perform_SR(self):
        # generate images
        generated_images = self.framework.generate_images(self.sample_images)
        self.save_images(generated_images)

    # saves images to the output path
    def save_images(self, images):
        dir_path = os.path.join(self.output_path, 'performer_output')
        os.makedirs(dir_path, exist_ok=True)

        for i in range(len(images)):
            img = cv2.cvtColor(np.array(images[i])*255, cv2.COLOR_RGB2BGR)
            path = os.path.join(dir_path, 'image_' + f"{i+1:03d}" + '.jpg')
            
            cv2.imwrite(path, img)