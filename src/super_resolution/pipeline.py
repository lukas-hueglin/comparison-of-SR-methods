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

from abc import ABC

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

def get_next_version(raw_path):
    index = 2
    output_path = raw_path + '_v.' + f"{1:03d}"

    # check if the path is already used and count up
    while os.path.exists(output_path):
        output_path = raw_path + '_v.' f"{index:03d}"
        index += 1

    # create this folder
    os.makedirs(output_path, exist_ok=True)

    return output_path


## Pipeline class ##
class Pipeline(ABC):
    def __init__(self, framework=None):
        self.framework = framework

        self.model_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, 'models'))
        self.output_path = get_next_version(os.path.join(self.model_path, self.framework.name))

    ## setter functions for the class variables.
    def set_framework(self, framework):
        self.framework = framework


    def load_framework(self, framework):
        self.output_path = os.path.join(self.model_path, framework)
        file_path = os.path.join(self.output_path,'checkpoints', 'saved.ckpt')

        with open(file_path, 'rb') as file:
            variables = dill.load(file)

        self.framework = variables['class']()
        self.framework.load_variables(variables)

    # This function is used to create the ABOUT.md file
    def create_ABOUT_file(self):
        # get info from framework
        text = self.framework.get_info()
        # add 3 - Training parameters
        text += '## 3 - Training parameters\n\n'
        text += 'Date: ' + str(datetime.date.today()) + '\n\n'
        text += 'Epochs: ' + str(self.epochs) + '</br>\n'
        text += 'Batch size: ...' + '</br>\n'
        text += 'Buffer size: ...' + '\n\n'
        # add 4 - datasets (in future)
        text += '## 4 - Datasets\n\n'
        text += 'Dataset: ... </br>\n'
        text += 'Training - Validation ratio: ...\n\n'
        # add 5 - Sample images
        text += '## 6 - Sample Images\n\n'
        text += '*Note: All these images are available under [progress images](./progress_images/)*\n\n'

        image_path = os.path.join(self.output_path, 'progress_images')
        image_dirs = os.listdir(image_path)
        for image in image_dirs:
            text += '![' + image + '](./progress_images/' + image + '/animfile.gif)\n'

        # write file
        file = open(self.output_path+'\\ABOUT.md', 'w')
        file.write(text)
        file.close()



class Trainer(Pipeline):
    def __init__(self, framework=None, dataset_loader=None, sample_loader=None, epochs=1):
        super().__init__(framework)

        self.dataset_loader = dataset_loader

        self.sample_loader = sample_loader
        self.sample_images = self.sample_loader.load_samples()

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
            for epoch in range(self.epochs):  
                # Add epoch timer
                epoch_time = time.perf_counter()

                # print a message
                print(TColors.HEADER + '\nEpoch ' + str(epoch) + ':\n' +TColors.ENDC)

                # prepare the data loading process
                self.dataset_loader.prepare_loading(train=True)

                # iterate over each batch
                num_batches = int(np.ceil(self.dataset_loader.train_size/self.dataset_loader.batch_size))
                for batch in tqdm(range(num_batches)):
                    # get data to train
                    (features, feature_time), (labels, label_time) = self.dataset_loader.access_loading()

                    # train
                    now = time.perf_counter()
                    generated_images, loss = self.framework.train_step(features, labels)

                    # stop timer
                    train_time = time.perf_counter() - now

                    # generate sys_load
                    cpu_load = psutil.cpu_percent(interval=None, percpu=False)
                    ram_load = psutil.virtual_memory().percent
                    gpu_load = []
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        gpu_load.append(gpu.load * 100)

                    # add values to stats recorder
                    self.framework.update_stats_recorder(
                        loss=loss,
                        sys_load=(cpu_load, ram_load, gpu_load),
                        time=(feature_time, label_time, train_time, None),
                        metrics=(generated_images, labels)
                    )

                # join the processes
                self.dataset_loader.close_loading()

                # Generate sample image
                if self.output_path != None and self.sample_images != None:
                    self.perform_SR(epoch)

                # Add epoch time
                self.framework.update_stats_recorder(
                    time=(None, None, None, time.perf_counter() - epoch_time),
                )

            # plot stats
            plot_path = os.path.join(self.output_path, 'statistics')
            name = self.framework.name
            self.framework.plot_and_save_stats(plot_path, self.epochs, name)

            # create gif
            image_path = image_path = os.path.join(self.output_path, 'progress_images')
            create_gif(image_path)

            # make ABOUT.md file
            self.create_ABOUT_file()

            # save framework
            self.save_framework()

        else:
            print(TColors.WARNING + 'The training dataset or the framework is not specified!' + TColors.ENDC)

    def perform_SR(self, epoch):
        # generate images
        generated_images = self.framework.generate_images(self.sample_images)
        self.save_sample_images(generated_images, epoch)

    def save_sample_images(self, images, epoch):
        for i in range(len(images)):
            img = cv2.cvtColor(np.array(images[i])*255, cv2.COLOR_RGB2BGR)

            # make paths
            dir_path = os.path.join(self.output_path, 'progress_images', 'image_' + f"{i:02d}")
            img_path = os.path.join(dir_path, 'epoch_' + f"{epoch:03d}" + '.jpg')

            # make a new folder
            os.makedirs(dir_path, exist_ok=True)
            
            cv2.imwrite(img_path, img)

    def save_framework(self):
        variables = self.framework.save_variables()

        ckpt_path = os.path.join(self.output_path, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)

        file_path = os.path.join(ckpt_path, 'saved.ckpt')
        with open(file_path, 'wb') as file:
            dill.dump(variables, file)
    
class Validator(Pipeline):
    def __init__(self, framework=None, dataset_loader=None):
        super().__init__(framework, dataset_loader)

        self.dataset_loader = dataset_loader

    ## setter functions for the class variables.
    def set_dataset_loader(self, dataset_loader):
        self.dataset_loader = dataset_loader


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

            # train
            now = time.perf_counter()
            generated_images, loss = self.framework.train_step(features, labels)

            # stop timer
            train_time = time.perf_counter() - now

            # generate sys_load
            cpu_load = psutil.cpu_percent(interval=None, percpu=False)
            ram_load = psutil.virtual_memory().percent
            gpu_load = []
            gpus = GPUtil.getGPUs()
            for gpu in gpus:
                gpu_load.append(gpu.load * 100)

            # add values to stats recorder
            self.framework.update_stats_recorder(
                loss=loss,
                sys_load=(cpu_load, ram_load, gpu_load),
                time=(feature_time, label_time, train_time),
                metrics=(generated_images, labels)
            )

        # join the processes
        self.dataset_loader.close_loading()

        # plot stats
        plot_path = os.path.join(self.output_path, 'statistics')
        name = self.framework.name
        self.framework.plot_and_save_stats(plot_path, self.epochs, name)

        # save framework
        self.serialize()

    def save_framework(self):
        variables = self.framework.save_variables()

        ckpt_path = os.path.join(self.output_path, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)

        file_path = os.path.join(ckpt_path, 'saved.ckpt')
        with open(file_path, 'wb') as file:
            dill.dump(variables, file)


class Performer(Pipeline):
    def __init__(self, framework=None, sample_loader=None, output_path=None):
        super().__init__(framework)

        self.sample_loader = sample_loader
        self.sample_images = self.sample_loader.load_samples()

        self.output_path = output_path

     ## setter functions for the class variables.
    def set_sample_loader(self, sample_loader):
        self.sample_loader = sample_loader
        self.sample_images = self.sample_loader.load_samples()

    def set_output_path(self, output_path):
        self.output_path = output_path


    def perform_SR(self):
        # generate images
        generated_images = self.framework.generate_images(self.sample_images)
        self.save_images(generated_images)

    def save_images(self, images):
        for i in range(len(images)):
            img = cv2.cvtColor(np.array(images[i])*255, cv2.COLOR_RGB2BGR)
            path = os.path.join(self.output_path, 'image_' + f"{i:03d}" + '.jpg')
            
            cv2.imwrite(path, img)