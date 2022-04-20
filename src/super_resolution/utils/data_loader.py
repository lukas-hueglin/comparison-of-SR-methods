### data_loader.py ###

import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from terminal_output import TColors

## helper functions ##

def visitor_func(name, node):
    if isinstance(node, h5py.Dataset):
        return name
        

## DatasetLoader class ##
class DatasetLoader():
    def __init__(self, path=None, feature_lod = None, label_lod = None, ds_batch_size = None, ds_num_batches = None, train_ratio = 0.8):
        self.path = path
        self.feature_lod = feature_lod
        self.label_lod = label_lod

        self.ds_batch_size = ds_batch_size
        self.ds_num_batches = ds_num_batches

        self.training_dataset = None
        self.validation_dataset = None

    def set_path(self, path):
        self.path = path

    def set_LODs(self, feature_lod, label_lod):
        self.feature_lod = feature_lod
        self.label_lod = label_lod

    def set_ds_batch_info(self, ds_batch_size, ds_num_batches):
        self.ds_batch_size = ds_batch_size
        self.ds_num_batches = ds_num_batches

    def load_images(self, image_range=(0, -1)):
        if (self.path != None and self.feature_lod != None and self.label_lod != None
                and self.ds_batch_size != None and self.ds_num_batches != None):

                    # change batch_range if there was no custom entry made
            if image_range[0] > image_range[1]:
                image_range = (0, self.ds_num_batches * self.ds_batch_size)

            feature_path = os.path.join(self.path, 'data', 'LOD_' + str(self.feature_lod) + '.hdf5')
            label_path = os.path.join(self.path, 'data', 'LOD_' + str(self.label_lod) + '.hdf5')

            features = []
            labels = []

            print(TColors.OKBLUE + 'Loading featues from ' + feature_path + ':\n' + TColors.ENDC)
            with h5py.File(feature_path, 'r') as hf:
                try:
                    image_count = 0
                    batches = hf.keys()
                    for b in tqdm(batches):
                        images = hf[b].keys()
                        for i in images:
                            if image_count >= image_range[0]:
                                features.append(np.array(hf[b+'/'+i]))
                            if image_count >= image_range[1]-1:
                                raise StopIteration
                            image_count += 1
                except StopIteration:
                    pass

            print(TColors.OKBLUE + '\n\nLoading labels from ' + label_path + ':\n' + TColors.ENDC)
            with h5py.File(label_path, 'r') as hf:
                try:
                    image_count = 0
                    batches = hf.keys()
                    for b in tqdm(batches):
                        images = hf[b].keys()
                        for i in images:
                            if image_count >= image_range[0]:
                                labels.append(np.array(hf[b+'/'+i]))
                            if image_count >= image_range[1]-1:
                                raise StopIteration
                            image_count += 1
                except StopIteration:
                    pass

            return np.array(features), np.array(labels)

        else:
            print(TColors.WARNING + 'Some variables are not specified!' + TColors.ENDC)