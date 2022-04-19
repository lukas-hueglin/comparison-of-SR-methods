### data_loader.py ###

import os
import cv2
import numpy as np
from tqdm import tqdm

from terminal_output import TColors

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

    def load_features(self):
        if self.path is not None and self.feature_lod is not None:
            data_path = os.path.join(self.path, 'data', 'LOD_' + str(self.feature_lod))
            images = []

            for path, subdirs, files in os.walk(data_path):
                for name in files:
                    images.append(cv2.imread(os.path.join(path, name)))

            return images

        else:
            print(TColors.WARNING + 'Path and feature_lod is not specified!' + TColors.ENDC)

    def load_labels(self):
        if self.path is not None and self.feature_lod is not None:
            data_path = os.path.join(self.path, 'data', 'LOD_' + str(self.feature_lod))
            images = []

            for path, subdirs, files in os.walk(data_path):
                for name in files:
                    images.append(cv2.imread(os.path.join(path, name)))

            return images

        else:
            print(TColors.WARNING + 'Path and feature_lod is not specified!' + TColors.ENDC)
