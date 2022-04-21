### data_loader.py ###

from cProfile import label
import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm

from terminal_output import TColors

## helper functions ##



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

    def load_images(self, first_image = 0, num_images = -1):
        if (self.path != None and self.feature_lod != None and self.label_lod != None
                and self.ds_batch_size != None and self.ds_num_batches != None):

            # change batch_range if there was no custom entry made
            if num_images <= -1:
                num_images = self.ds_num_batches * self.ds_batch_size

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
                            if image_count >= first_image:
                                features.append(np.array(hf[b+'/'+i]))
                                image_count += 1
                            if image_count >= num_images-1:
                                raise StopIteration
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
                            if image_count >= first_image:
                                labels.append(np.array(hf[b+'/'+i]))
                                image_count += 1
                            if image_count >= num_images-1:
                                raise StopIteration
                except StopIteration:
                    pass

            return np.array(features), np.array(labels)

        else:
            print(TColors.WARNING + 'Some variables are not specified!' + TColors.ENDC)

dataset_loader = DatasetLoader(
    path='D:\Local UNSPLASH Dataset Full',
    feature_lod=5,
    label_lod=4,
    ds_batch_size=1000,
    ds_num_batches=100
)

features, lables = dataset_loader.load_images()

plt.imshow(features[0])
plt.show()
plt.imshow(lables[0])
plt.show()
print(features.shape, lables.shape)