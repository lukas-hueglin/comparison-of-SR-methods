### pipeline.py ###
# This module connects the framework with the data. It
# is a container for all it needs to train or test out a framework.
##

import tensorflow as tf
import multiprocessing

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

from utils import TColors


## helper_functions ##

# This function is called every epoch and saves the predicted images of the network.
def generate_and_save(path, epoch,  images, gen_func):
    # generate the output with the gen_func (generate_images() in Method class)
    generated_images = gen_func(images)

    # save each image
    for i in range(len(generated_images)):
        img_path = os.path.join(path,  'image_' + str(i))

        # make a new folder
        os.makedirs(img_path, exist_ok=True)

        # make name
        name = os.path.join(img_path, 'epoch_' + f'{epoch:03d}' + '.jpg')
        
        # the images have to be converted to BGR and streched to 255
        img = cv2.cvtColor(np.array(generated_images[i])*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, img)

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

## Pipeline class ##
class Pipeline():
    def __init__(self, framework = None, epochs = 10, epoch_start=0, dataset_loader = None, path=None,  sample_images = None):
        
        self.framework = framework
        self.epochs = epochs
        self.epoch_start = epoch_start

        self.dataset_loader = dataset_loader

        # Set the path of the framework root folder
        rel_path = os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, 'models', self.framework.name)
        raw_path = os.path.abspath(rel_path)
        index = 1

        # check if the path is already used and count up
        if path is None:
            self.path = raw_path + '_v.' + f"{index:02d}"
            while os.path.exists(self.path):
                self.path = raw_path + '_v.' f"{index:02d}"
                index += 1

            # create this folder
            os.makedirs(self.path, exist_ok=True)
        else:
            self.path = path

        self.sample_images = sample_images

    ## setter functions for the class variables.
    def set_path(self, path):
        self.path = path

    def set_framework(self, framework):
        self.framework = framework

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_train_data(self, training_data):
        self.training_data = training_data

    def set_validation_data(self, validation_data):
        self.validation_data = validation_data

    def set_sample_images(self, sample_images):
        self.sample_images = sample_images


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

        image_path = os.path.join(self.path, 'progress_images')
        image_dirs = os.listdir(image_path)
        for image in image_dirs:
            text += '![' + image + '](./progress_images/' + image + '/animfile.gif)\n'

        # write file
        file = open(self.path+'\\ABOUT.md', 'a')
        file.write(text)
        file.close()


    # The main train function. This function gets called by the user.
    def train(self):
        # check if everything is specified
        if self.dataset_loader != None:

            # iterate over each epoch
            for epoch in range(self.epoch_start, self.epochs+self.epoch_start):  
                # print a message
                print(TColors.HEADER + '\nEpoch ' + str(epoch) + ':\n' +TColors.ENDC)

                # start with multiprocessing
                # (with help from: https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce)
                feature_queue = multiprocessing.Queue()
                label_queue = multiprocessing.Queue()

                feature_loader = multiprocessing.Process(
                    target=self.dataset_loader.load_dataset_mp,
                    args=(feature_queue, True)
                )
                label_loader = multiprocessing.Process(
                    target=self.dataset_loader.load_dataset_mp,
                    args=(label_queue, False)
                )

                # start
                f_start_timer = time.perf_counter()
                l_start_timer = time.perf_counter()

                feature_loader.start()
                label_loader.start()

                # iterate over each batch
                num_batches = int(np.ceil(self.dataset_loader.train_size/self.dataset_loader.batch_size))
                for batch in tqdm(range(num_batches)):
                    # wait until batch arrives
                    feature_is_loaded = False
                    label_is_loaded = False
                    while True:
                        # check if features are ready
                        if not feature_queue.empty():
                            if not feature_is_loaded:
                                # add load time to stats recorder
                                now = time.perf_counter()
                                self.framework.update_stats_recorder(time=(now-f_start_timer, None, None))
                                f_start_timer = now
                            feature_is_loaded = True

                        # check if labels are ready
                        if not label_queue.empty():
                            if not label_is_loaded:
                                # add load time to stats recorder
                                now = time.perf_counter()
                                self.framework.update_stats_recorder(time=(None, now-l_start_timer, None))
                                l_start_timer = now
                            label_is_loaded = True

                        if feature_is_loaded and label_is_loaded:
                            # load the data into tensors
                            features = tf.convert_to_tensor(feature_queue.get())
                            labels = tf.convert_to_tensor(label_queue.get())
                            break
                        time.sleep(0.01)

                    # train
                    train_start_timer = time.perf_counter()
                    generated_images, loss = self.framework.train_step(features, labels)

                    # add train time to stats recorder
                    now = time.perf_counter()
                    self.framework.update_stats_recorder(time=(None, None, now-train_start_timer))

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
                        metrics=(generated_images, labels)
                    )

                # join the processes
                feature_loader.join()
                label_loader.join()

                # Generate sample image
                if self.path != None and self.sample_images != None:
                    image_path = os.path.join(self.path, 'progress_images')
                    generate_and_save(image_path, epoch, self.sample_images, self.framework.generate_images)

            # plot stats
            plot_path = os.path.join(self.path, 'statistics')
            name = self.framework.name
            self.framework.plot_and_save_stats(plot_path, self.epochs + self.epoch_start, name)

            # create gif
            image_path = image_path = os.path.join(self.path, 'progress_images')
            create_gif(image_path)

            # make ABOUT.md file
            self.create_ABOUT_file()

            # save framework
            self.serialize()

        else:
            print(TColors.WARNING + 'The training dataset or the framework is not specified!' + TColors.ENDC)

    def serialize(self):
        variables = self.framework.save_variables()

        ckpt_path = os.path.join(self.path, 'checkpoints')
        os.makedirs(ckpt_path, exist_ok=True)

        file_path = os.path.join(ckpt_path, 'saved.ckpt')
        with open(file_path, 'wb') as file:
            dill.dump(variables, file)

    def deserialize(self):
        file_path = os.path.join(self.path,'checkpoints', 'saved.ckpt')
        with open(file_path, 'rb') as file:
            variables = dill.load(file)

        self.framework = variables['class']()
        self.framework.load_variables(variables)