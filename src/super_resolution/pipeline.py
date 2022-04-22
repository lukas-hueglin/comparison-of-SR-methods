### pipeline.py ###
# This module connects the framework with the data. It
# is a container for all it needs to train or test out a framework.
##

import os
import cv2
import numpy as np

from tqdm import tqdm

from utils import TColors


## helper_functions ##

# This function is called every epoch and saves the predicted images of the network.
def generate_and_save(path, images, gen_func):
    # generate the output with the gen_func (generate_images() in Method class)
    generated_images = gen_func(images)

    # make a new folder
    os.makedirs(path, exist_ok=True)

    # save each image
    for i in range(len(generated_images)):
        name = os.path.join(path,  'image_' + str(i) + '.jpg')
        
        # the images have to be converted to BGR and streched to 255
        img = cv2.cvtColor(np.array(generated_images[i])*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, img)


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
                    self.framework.train_step(features, labels)

                # Generate sample image
                if self.path != None and self.sample_images != None:
                    image_path = os.path.join(self.path, 'progress_images', 'epoch_' + str(epoch))
                    generate_and_save(image_path, self.sample_images, self.framework.generate_images)

        else:
            print(TColors.WARNING + 'The training dataset or the framework is not specified!' + TColors.ENDC)
