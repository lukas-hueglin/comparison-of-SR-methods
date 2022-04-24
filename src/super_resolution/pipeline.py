### pipeline.py ###
# This module connects the framework with the data. It
# is a container for all it needs to train or test out a framework.
##

import os
import datetime

import cv2
import imageio

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
        name = os.path.join(img_path, 'epoch_' + str(epoch) + '.jpg')
        
        # the images have to be converted to BGR and streched to 255
        img = cv2.cvtColor(np.array(generated_images[i])*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, img)

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
                image_path = os.path.join(dir_path, i)
                image = imageio.imread(image_path)
                writer.append_data(image)


## Pipeline class ##
class Pipeline():
    def __init__(self, framework = None, epochs = 10, training_data = None, validation_data = None, sample_images = None):
        
        self.framework = framework
        self.epochs = epochs

        self.training_data = training_data
        self.validation_data = validation_data

        # Set the path of the framework root folder
        rel_path = os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, 'models', self.framework.name)
        raw_path = os.path.abspath(rel_path)
        index = 1

        # check if the path is already used and count up
        self.path = raw_path + '_v.' + f"{index:02d}"
        while os.path.exists(self.path):
            self.path = raw_path + '_v.' f"{index:02d}"
            index += 1

        # create this folder
        os.makedirs(self.path)

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
        # helper function for adding a image
        def add_image(img_name):
            return '![' + img_name + '](./progress_images/' + img_name + '/animfile.gif)\n'

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
            text += add_image(image)

        # write file
        file = open(self.path+'\\ABOUT.md', 'a')
        file.write(text)
        file.close()


    # The main train function. This function gets called by the user.
    def train(self):
        # check if everything is specified
        if self.training_data != None and self.framework != None:

            # iterate over each epoch
            for epoch in range(self.epochs):  
                # print a message
                print(TColors.HEADER + '\nEpoch ' + str(epoch) + ':\n' +TColors.ENDC)

                # iterate over each batch
                for batch in tqdm(self.training_data):
                    features, labels = batch
                    generated_images, loss = self.framework.train_step(features, labels)

                    # add loss to StatsRecorder
                    self.framework.method.update_stats_recorder(loss)

                # Generate sample image
                if self.path != None and self.sample_images != None:
                    image_path = os.path.join(self.path, 'progress_images')
                    generate_and_save(image_path, epoch, self.sample_images, self.framework.generate_images)

            # plot stats
            plot_path = os.path.join(self.path, 'statistics')
            name = self.framework.name
            self.framework.method.save_stats(plot_path, self.epochs, name)

            # create gif
            image_path = image_path = os.path.join(self.path, 'progress_images')
            create_gif(image_path)

            # make ABOUT.md file
            self.create_ABOUT_file()

        else:
            print(TColors.WARNING + 'The training dataset or the framework is not specified!' + TColors.ENDC)
