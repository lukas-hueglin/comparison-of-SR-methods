### data_loader.py ###
# In this module, the DatasetLoader and the SampleLoader classes are defined.
# They both load images from a .hdf5 file. The main differences between these classes are:
#   - DatasetLoader: Loads Features and Labels into a tf.Dataset
#   - SampleLoader: Loads just one dataset into a tf.tensor
##

from cmath import inf
import tensorflow as tf

import os
import numpy as np
import h5py
from tqdm import tqdm
from enum import Enum

from utils import TColors


## helper enum
class DatasetType(Enum):
    SUPERVISED = 1
    UNSUPERVISED = 2


## helper functions ##

# loads images from a .hdf5 file into a numpy array
def load_images(path, first_image = 0, num_images = -1, silent=False):
        imgs = []

        # print message
        if not silent:
            print(TColors.OKBLUE + 'Loading images from ' + path + ':\n' + TColors.ENDC)
        
        # open .hdf5 file
        with h5py.File(path, 'r') as hf:
            # do try-except to stop double loop
            try:
                image_count = 0
                batches = hf.keys()

                # iterate all batches
                if silent:
                    it = batches
                else:
                    it = tqdm(batches)
                for b in it:
                    images = hf[b].keys()

                    # iterate all images
                    for i in images:
                        # check if the image should be appended
                        if image_count >= first_image:
                            imgs.append(np.array(hf[b+'/'+i])/255) # normalize
                        if image_count - first_image >= num_images - 1:
                            raise StopIteration
                        # add one image
                        image_count += 1
            except StopIteration:
                pass

        return np.array(imgs)


# loads Features and Labels into a tf.Dataset
class DatasetLoader():
    def __init__(self, path=None, feature_lod = 1, label_lod = 0, batch_size = 20, buffer_size = 100, dataset_type = DatasetType.SUPERVISED, train_ratio = 0.8, dataset_size = -1):
        self.path = path
        self.feature_lod = feature_lod
        self.label_lod = label_lod

        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.dataset_type = dataset_type
        
        if dataset_size < 0:
            self.dataset_size = self.get_dataset_size()
        else:
            self.dataset_size = dataset_size

        self.train_size = int(self.dataset_size*train_ratio)

    ## setter functions for class variables
    def set_path(self, path):
        self.path = path

    def set_LODs(self, feature_lod, label_lod):
        self.feature_lod = feature_lod
        self.label_lod = label_lod

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
    
    def set_buffer_size(self, buffer_size):
        self.buffer_size = buffer_size


    ## dataset loader function
    # REMINDER: The datasets are not repeated for each epoch, I am not sure if this is going to be a problem!
    def load_dataset(self, first_image = 0, num_images = -1):
        if self.path != None:
            # create the paths for the feature and label .hdf5 files.
            feature_path = os.path.join(self.path, 'data', 'LOD_' + str(self.feature_lod) + '.hdf5')
            label_path = os.path.join(self.path, 'data', 'LOD_' + str(self.label_lod) + '.hdf5')

            # change image range if there was no custom or a stupid entry made
            if num_images <= -1:
                num_images = inf

            # load the images into numpy arrays
            features = load_images(feature_path, first_image, num_images)
            labels = load_images(label_path, first_image, num_images)

            # shuffle features so a unsupervised dataset is generated
            if self.dataset_type == DatasetType.UNSUPERVISED:
                features = np.random.shuffle(features)


            # split the datasets into test and validation datasets
            f_train = features[:int(features.shape[0]*self.train_ratio)]
            f_validation = features[int(features.shape[0]*self.train_ratio):]
            l_train = labels[:int(labels.shape[0]*self.train_ratio)]
            l_validation = labels[int(labels.shape[0]*self.train_ratio):]

            training_dataset = tf.data.Dataset.from_tensor_slices((f_train, l_train))
            validation_dataset = tf.data.Dataset.from_tensor_slices((f_validation, l_validation))

            # batch the training_dataset
            training_dataset = training_dataset.batch(self.batch_size)

            return training_dataset, validation_dataset

    # dataset loader function with multiprocessing
    def load_batch(self, batch, return_dict):
        if self.path != None:
            # create the paths for the feature and label .hdf5 files.
            feature_path = os.path.join(self.path, 'data', 'LOD_' + str(self.feature_lod) + '.hdf5')
            label_path = os.path.join(self.path, 'data', 'LOD_' + str(self.label_lod) + '.hdf5')

            # set next_image
            next_image = batch*self.batch_size

            # set image_num
            if next_image + self.batch_size <= self.train_size:
                image_num = self.batch_size
            else:
                image_num = self.train_size - next_image

            # load the images into numpy arrays
            features = load_images(feature_path, next_image, image_num, silent=True)
            labels = load_images(label_path, next_image, image_num, silent=True)

            # shuffle features so a unsupervised dataset is generated
            if self.dataset_type == DatasetType.UNSUPERVISED:
                features = np.random.shuffle(features)

            return_dict['features'] = features
            return_dict['labels'] = labels

    # function which returns the size of the given dataset
    def get_dataset_size(self):
        # set lod0 as file
        file = os.path.join(self.path, 'data', 'LOD_0.hdf5')

        # open .hdf5 file
        with h5py.File(file, 'r') as hf:
            # do try-except to stop double loop
            image_count = 0
            batches = hf.keys()

            # iterate all batches
            for b in batches:
                images = hf[b].keys()
                image_count += len(images)

        return image_count


# loads just one dataset into a tf.tensor
class SampleLoader():
    def __init__(self, path=None, lod = 1):
        self.path = path
        self.lod = lod

    ## setter functions for class variables
    def set_path(self, path):
        self.path = path

    def set_LOD(self, lod):
        self.lod = lod


    ## sample loader function
    def load_samples(self, first_image = 0, num_images = -1):
        if self.path != None:
            # create the paths for the feature and label .hdf5 files.
            path = os.path.join(self.path, 'data', 'LOD_' + str(self.lod) + '.hdf5')

            # change image range if there was no custom or a stupid entry made
            if num_images <= -1:
                num_images = inf

            # load the images into numpy arrays
            samples = load_images(path, first_image, num_images)

            # create tensor
            tensor = tf.convert_to_tensor(samples)

            return tensor