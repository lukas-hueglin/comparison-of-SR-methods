### data_loader.py ###
# In this module, the DatasetLoader and the SampleLoader classes are defined.
# They both load images from a .hdf5 file. The main differences between these classes are:
#   - DatasetLoader: Loads Features and Labels into a tf.Dataset
#   - SampleLoader: Loads just one dataset into a tf.tensor
##

import tensorflow as tf

import multiprocessing as mp
import time

import os
import numpy as np
import h5py

import cv2

from utils import TColors


## helper functions ##

# resizes an array of images to de desired squared size
def resize_images(imgs, size, interpolation):
    output_imgs = []

    for img in imgs:
        output_imgs.append(cv2.resize(img, (size, size), interpolation=interpolation))

    return np.array(output_imgs)


# loads Features and Labels into a tf.Dataset
class DatasetLoader():
    def __init__(self, path=None, feature_lod = 1, label_lod = 0, batch_size = 20, dataset_type = 'supervised', train_ratio = 0.8, dataset_size = -1):
        self.path = path
        self.feature_path = os.path.join(path, 'data', 'LOD_' + str(feature_lod) + '.hdf5')
        self.label_path = os.path.join(path, 'data', 'LOD_' + str(label_lod) + '.hdf5')

        self.batch_size = batch_size

        self.dataset_type = dataset_type
        
        if dataset_size < 0:
            self.dataset_size = self.get_dataset_size()
        else:
            self.dataset_size = dataset_size

        self.train_ratio = train_ratio
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


    # prepare the loading of the images
    def prepare_loading(self, train=True):
        # start with multiprocessing
        # (with help from: https://stackoverflow.com/questions/10415028/how-can-i-recover-the-return-value-of-a-function-passed-to-multiprocessing-proce)

        # create queues and jobs
        feature_queue = mp.Queue()
        label_queue = mp.Queue()

        feature_loader = mp.Process(
            target=self.load_dataset_mp,
            args=(feature_queue, self.feature_path, train, True)
        )
        label_loader = mp.Process(
            target=self.load_dataset_mp,
            args=(label_queue, self.label_path, train, False)
        )

        # start processes and add them to the class variables
        feature_loader.start()
        label_loader.start()

        self.feature_job = (feature_loader, feature_queue)
        self.label_job = (label_loader, label_queue)

    # with this function the pipeline can access the images while loading
    def access_loading(self):
        # wait until batch arrives
        while True:
            if not self.feature_job[1].empty() and not self.label_job[1].empty():
                return (self.feature_job[1].get(), self.label_job[1].get())
            time.sleep(0.01)

    # close the multiprocessing jobs
    def close_loading(self):
        self.feature_job[0].join()
        self.label_job[0].join()

        self.feature_job = (None, None)
        self.label_job = (None, None)

    # dataset loader function with multiprocessing
    def load_dataset_mp(self, queue, path, train=True, features=True):
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
                        if features:
                            array = np.array(hf[b+'/'+i])/255 # normalize
                        else:
                            array = np.array(hf[b+'/'+i])/127.5 - 1 # normalize

                        images.append(array) 
                        image_count += 1

                        # check if batch is full
                        if image_count % self.batch_size == 0:
                            # shuffle features so a unsupervised dataset is generated
                            if self.dataset_type == 'unsupervised':
                                np.random.shuffle(images)

                            # stop timer
                            now = time.perf_counter()

                            # put on queue
                            queue.put((images,now-timer), block=True, timeout=None)

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
                if self.dataset_type == 'unsupervised':
                    np.random.shuffle(images)

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
    def __init__(self, path=None, resolution=32, batch_size=20):
        self.path = path
        self.resolution = resolution


        self.batch_size = batch_size
        self.size = self.get_dataset_size()

    ## setter functions for class variables
    def set_path(self, path):
        self.path = path

    def set_resolution(self, resolution):
        self.resolution = resolution


    ## sample loader function
    def load_samples(self):
        if self.path != None:
            images = []

            for path, subdirs, files in os.walk(self.path):
                for name in files:
                    images.append(cv2.imread(os.path.join(self.path, name))[:, :, [2, 1, 0]])

            # create tensor
            resized_images = resize_images(images, self.resolution, cv2.INTER_LANCZOS4) / 255
            tensor = tf.convert_to_tensor(resized_images)

            return tensor
        else:
            print(TColors.WARNING + 'The Sample Loader path is not specified!' + TColors.ENDC)

    # function which returns the size of the given dataset
    def get_dataset_size(self):
        # set lod0 as file
        path = os.path.join(self.path)

        image_count = 0
        for path, subdirs, files in os.walk(path):
            image_count += len(files)

        return image_count