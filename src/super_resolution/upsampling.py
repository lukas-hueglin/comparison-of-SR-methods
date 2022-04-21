### upsampling.py ###

import tensorflow as tf
import cv2
import numpy as np
import sys
from abc import ABC, abstractmethod

from utils import TColors


## upsampling functions ##
def bicubic(img, res):
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_CUBIC)

def bilinear(img, res):
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_LINEAR)

def lanczos(img, res):
    return cv2.resize(img, (res, res), interpolation=cv2.INTER_LANCZOS4)

## Framework classes ##
class Framework(ABC):
    def __init__(self, input_res = None, output_res = None, upsample_function = None):

        self.input_res = input_res
        self.output_res = output_res

        self.upsample_function = upsample_function

        self.check_status()

    def set_resolutions(self, input_res, output_res):
        self.input_res = input_res
        self.output_res = output_res
        self.check_status()

    def set_upsample_function(self, upsample_function):
        self.upsample_function = upsample_function
    
    def check_status(self):
        if self.input_res is not None and self.output_res is not None:
            q = float(np.log2(self.output_res/self.input_res))

            if q.is_integer() is False and q == self.steps:
                print(TColors.FAIL + 'The input and output resolutions are not a power of 2' + TColors.ENDC)
                sys.exit()
        

    @abstractmethod
    def train_step(self, feature, label):
        pass


class PreUpsampling(Framework):
    def __init__(self, input_res=None, output_res=None, upsample_function=None, method=None):
        super().__init__(input_res, output_res, upsample_function)

        self.method = method

    def set_method(self, method):
        self.method = method


    @tf.function
    def train_step(self, features, labels):
        upsampled_features = []
        
        for f in features:
                upsampled_features.append(self.upsample_function(f, self.output_res))

        self.method.train_method(upsampled_features, labels)


class ProgressiveUpsampling(Framework):
    def __init__(self, input_res=None, output_res=None, method=None, upsample_function=None, steps = 1):
        super().__init__(input_res, output_res, upsample_function)

        self.methods = [method] * steps
        self.steps = steps

    def set_method(self, method):
        self.methods = [method] * self.steps

    def set_steps(self, steps):
        self.steps = steps
        self.methods = [self.methods[0]] * self.steps
        self.check_status()


    @tf.function
    def train_step(self, features, labels):
        for i in range(self.steps):
            res = self.input_res * np.power(2, i)

            upsampled_features = []
            downsampled_labels = []

            for f in features:
                upsampled_features.append(self.upsample_function(f, res))

            for l in labels:
                downsampled_labels.append(lanczos(l, res)) # using lanczos because it's a good downsample method

            features = self.method.train_method(upsampled_features, downsampled_labels)
