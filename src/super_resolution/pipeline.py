### pipeline.py ###

from tqdm import tqdm
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np

from utils import TColors
from utils import DatasetLoader
from upsampling import Framework

import tensorflow as tf


## helper enums ##

class Tasks(Enum):
    TRAIN = 1
    VALIDATE = 2

## helper_functions ##

## Pipeline class ##
class Pipeline():
    def __init__(self, framework = None, epochs = 10, training_data = None, validation_data = None, sample_image = None, tasks = None):
        self.framework = framework
        self.epochs = epochs

        self.training_data = training_data
        self.validation_data = validation_data

        self.sample_image = sample_image
        self.tasks = tasks

    def set_framework(self, framework):
        self.framework = framework

    def set_epochs(self, epochs):
        self.epochs = epochs

    def set_train_data(self, training_data):
        self.training_data = training_data

    def set_validation_data(self, validation_data):
        self.validation_data = validation_data

    def set_tasks(self, tasks):
        self.tasks = tasks

    def train(self):
        if self.training_data != None and self.framework != None:
            for epoch in tqdm(range(self.epochs)):
                for batch in self.training_data:
                    self.framework.train_steps(batch[0], batch[1])
        else:
            print(TColors.WARNING + 'The training dataset or the framework is not specified!' + TColors.ENDC)
