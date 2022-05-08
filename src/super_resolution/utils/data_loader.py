### data_loader.py ###
# In this module, the DatasetLoader and the SampleLoader classes are defined.
# They both load images from a .hdf5 file. The main differences between these classes are:
#   - DatasetLoader: Loads Features and Labels into a tf.Dataset
#   - SampleLoader: Loads just one dataset into a tf.tensor
##

from cmath import inf
import tensorflow as tf

import multiprocessing
import time

import os
import numpy as np
import h5py
from tqdm import tqdm
from enum import Enum

from utils import TColors


## helper enum
class DatasetType(Enum):
    SUPERVISED = 0
    UNSUPERVISED = 1

class DatasetPortion(Enum):
    TRAIN_FEATURE = 0
    TRAIN_LABEL = 1
    VALIDATION_FEATURE = 2
    VALIDATION_LABEL = 3


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
        self.feature_path = os.path.join(path, 'data', 'LOD_' + str(feature_lod) + '.hdf5')
        self.label_path = os.path.join(path, 'data', 'LOD_' + str(label_lod) + '.hdf5')

        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.dataset_type = dataset_type
        
        if dataset_size < 0:
            self.dataset_size = self.get_dataset_size()
        else:
            self.dataset_size = dataset_size

        self.train_size = int(self.dataset_size*train_ratio)
        self.validation_size = self.dataset_size - self.train_size

        self.feature_job = (None, None)
        self.label_job = (None, None)

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


    # prepare dataset_loader 
    def prepare_loading(self, train=True):
        # start with multiprocessing
        # (with help from: https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce)

        # create queues and jobs
        feature_queue = multiprocessing.Queue()
        label_queue = multiprocessing.Queue()

        feature_loader = multiprocessing.Process(
            target=self.load_dataset_mp,
            args=(feature_queue, self.feature_path, train)
        )
        label_loader = multiprocessing.Process(
            target=self.load_dataset_mp,
            args=(label_queue, self.label_path, train)
        )

        # start processes and add them to the class variables
        feature_loader.start()
        label_loader.start()

        self.feature_job = (feature_loader, feature_queue)
        self.label_job = (label_loader, label_queue)

    def access_loading(self):
        # wait until batch arrives
        while True:
            if not self.feature_job[1].empty() and not self.label_job[1].empty():
                return (self.feature_job[1].get(), self.label_job[1].get())
            time.sleep(0.01)

    def close_loading(self):
        self.feature_job[0].join()
        self.label_job[0].join()

        self.feature_job = (None, None)
        self.label_job = (None, None)

    # dataset loader function with multiprocessing
    def load_dataset_mp(self, queue, path, train=True):
        # start timer
        timer = time.perf_counter()

        # load the images
        images = []

        # set size
        size = self.train_size if train else self.validation_size
        
        # open .hdf5 file
        with h5py.File(path, 'r') as hf:
            # do try-except to stop double loop
            try:
                batch_nodes = hf.keys() if train else reversed(hf.keys())
                image_count = 0

                # iterate all batches
                for b in batch_nodes:
                    image_nodes = hf[b].keys() if train else reversed(hf[b].keys())

                    # iterate all images
                    for i in image_nodes:
                        images.append(np.array(hf[b+'/'+i])/255) # normalize
                        image_count += 1

                        # check if batch is full
                        if image_count % self.batch_size == 0:
                            # shuffle features so a unsupervised dataset is generated
                            if self.dataset_type == DatasetType.UNSUPERVISED:
                                images = np.random.shuffle(images)

                            # stop timer
                            now = time.perf_counter()

                            # put on queue
                            queue.put((images,now-timer))

                            timer = now
                            images = []

                        # check if finished
                        if image_count >= size:
                            raise StopIteration
            except StopIteration:
                pass

            # if images are left put them also on the queue
            if len(images) != 0:
                # shuffle features so a unsupervised dataset is generated
                if self.dataset_type == DatasetType.UNSUPERVISED:
                    images = np.random.shuffle(images)

                # put on queue
                queue.put((images, time.perf_counter()-timer))

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
    def __init__(self, path=None, lod = 1, batch_size=None):
        self.path = path
        self.lod = lod


        self.batch_size = batch_size
        self.size = self.get_dataset_size()

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