### pipeline.py ###

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils import TColors


## helper_functions ##

def generate_and_save(path, images, gen_func):
    generated_images = gen_func(images)

    os.makedirs(path, exist_ok=True)
    for i in range(len(generated_images)):
        name = os.path.join(path,  'image_' + str(i) + '.jpg')
        img = cv2.cvtColor(np.array(generated_images[i])*255, cv2.COLOR_RGB2BGR)
        cv2.imwrite(name, img)


## Pipeline class ##
class Pipeline():
    def __init__(self, framework = None, epochs = 10, training_data = None, validation_data = None, sample_images = None):
        
        self.framework = framework
        self.epochs = epochs

        self.training_data = training_data
        self.validation_data = validation_data

        # set path
        rel_path = os.path.join(os.path.dirname( __file__ ), os.pardir, os.pardir, 'models', self.framework.name)
        raw_path = os.path.abspath(rel_path)
        index = 1

        self.path = raw_path + '_v.' + f"{index:02d}"
        while os.path.exists(self.path):
            self.path = raw_path + '_v.' f"{index:02d}"
            index += 1

        self.sample_images = sample_images


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


    def train(self):
        if self.training_data != None and self.framework != None:
            for epoch in range(self.epochs):  
                print(TColors.HEADER + '\nEpoch ' + str(epoch) + ':\n' +TColors.ENDC)
                for batch in tqdm(self.training_data):
                    features, labels = batch
                    self.framework.train_step(features, labels)

                # Generate sample image
                if self.path != None and self.sample_images != None:
                    image_path = os.path.join(self.path, 'progress_images', 'epoch_' + str(epoch))
                    generate_and_save(image_path, self.sample_images, self.framework.generate_images)

        else:
            print(TColors.WARNING + 'The training dataset or the framework is not specified!' + TColors.ENDC)
