### data_loader.py ###

import tensorflow as tf

import os
import numpy as np
import h5py
from tqdm import tqdm

from terminal_output import TColors

## helper functions ##

def load_images(path, first_image = 0, num_images = -1):
        images = []

        print(TColors.OKBLUE + 'Loading images from ' + path + ':\n' + TColors.ENDC)
        with h5py.File(path, 'r') as hf:
            try:
                image_count = 0
                batches = hf.keys()
                for b in tqdm(batches):
                    images = hf[b].keys()
                    for i in images:
                        if image_count >= first_image:
                            images.append(np.array(hf[b+'/'+i]))
                            image_count += 1
                        if image_count >= num_images-1:
                            raise StopIteration
            except StopIteration:
                pass

        return np.array(images)

## DatasetLoader class ##
class DatasetLoader():
    def __init__(self, path=None, feature_lod = 1, label_lod = 0, batch_size = 20, buffer_size = 100,
                ds_batch_size = None, ds_num_batches = None, train_ratio = 0.8):
        self.path = path
        self.feature_lod = feature_lod
        self.label_lod = label_lod

        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.ds_batch_size = ds_batch_size
        self.ds_num_batches = ds_num_batches

        self.train_ratio = train_ratio

        self.training_dataset = None
        self.validation_dataset = None

    def set_path(self, path):
        self.path = path

    def set_LODs(self, feature_lod, label_lod):
        self.feature_lod = feature_lod
        self.label_lod = label_lod

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size

    def set_ds_batch_info(self, ds_batch_size, ds_num_batches):
        self.ds_batch_size = ds_batch_size
        self.ds_num_batches = ds_num_batches

    def load_dataset(self, first_image = 0, num_images = -1):
        if (self.path != None and self.feature_lod != None and self.label_lod != None
                and self.ds_batch_size != None and self.ds_num_batches != None):

            feature_path = os.path.join(self.path, 'data', 'LOD_' + str(self.feature_lod) + '.hdf5')
            label_path = os.path.join(self.path, 'data', 'LOD_' + str(self.label_lod) + '.hdf5')

            # change image range if there was no custom entry made
            if num_images <= -1:
                num_images = self.ds_num_batches * self.ds_batch_size

            features = load_images(feature_path, first_image, num_images)
            lables = load_images(label_path, first_image, num_images)

            f_train, f_validation = np.split(features, int(features.shape[0]*self.train_ratio))
            l_train, l_validation = np.split(lables, int(lables.shape[0]*self.train_ratio))

            train_dataset = tf.data.Dataset.from_tensor_slices((f_train, l_train))
            validation_dataset = tf.data.Dataset.from_tensor_slices((f_validation, l_validation))

            return train_dataset, validation_dataset

        else:
            print(TColors.WARNING + 'Some variables are not specified!' + TColors.ENDC)
        
        
    # REMINDER: The dataset is not repeated for each epoch, I am not sure if this is going to be a problem!

    # The supervised dataset is just batched
    def load_supervised_dataset(self, first_image = 0, num_images = -1):
        train_dataset, validation_dataset = self.load_dataset(first_image=first_image, num_images=num_images)

        train_dataset = train_dataset.batch(self.batch_size)

        return train_dataset, validation_dataset

    # The unsupervised dataset is also shuffled
    def load_unsupervised_dataset(self, first_image = 0, num_images = -1):
        train_dataset, validation_dataset = self.load_dataset(first_image=first_image, num_images=num_images)

        train_dataset = train_dataset.shuffle(buffer_size=self.buffer_size)
        train_dataset = train_dataset.batch(self.batch_size)

        return train_dataset, validation_dataset