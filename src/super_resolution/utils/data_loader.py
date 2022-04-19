### data_loader.py ###

import os
import cv2
import numpy as np
import h5py
import matplotlib.pyplot as plt

from terminal_output import TColors

## helper functions ##

def visitor_func(name, node):
    if isinstance(node, h5py.Dataset):
        return name
        

## DatasetLoader class ##
class DatasetLoader():
    def __init__(self, path=None, feature_lod = None, label_lod = None, train_ratio = 0.8):
        self.path = path
        self.feature_lod = feature_lod
        self.label_lod = label_lod

        self.training_dataset = None
        self.validation_dataset = None

    def set_path(self, path):
        self.path = path

    def set_LODs(self, feature_lod, label_lod):
        self.feature_lod = feature_lod
        self.label_lod = label_lod

    def load_images(self):
        if self.path != None and self.feature_lod != None and self.label_lod != None:
            feature_path = os.path.join(self.path, 'data', 'LOD_' + str(self.feature_lod) + '.hdf5')
            label_path = os.path.join(self.path, 'data', 'LOD_' + str(self.label_lod) + '.hdf5')

            features = []
            labels = []

            with h5py.File(feature_path, 'r') as hf:
                groups = hf.keys()
                for g in groups:
                    names = hf[g].keys()
                    for n in names:
                        features.append(np.array(hf[g + '/' +n]))

            with h5py.File(label_path, 'r') as hf:
                groups = hf.keys()
                for g in groups:
                    names = hf[g].keys()
                    for n in names:
                        labels.append(np.array(hf[g + '/' +n]))

            return np.array(features), np.array(labels)

        else:
            print(TColors.WARNING + 'Path and feature_lod is not specified!' + TColors.ENDC)
