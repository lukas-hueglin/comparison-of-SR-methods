### main.py ###

import tensorflow as tf

import cv2
import matplotlib.pyplot as plt

from utils import DatasetLoader
from pipeline import Pipeline

import presets

## global parameters

EPOCHS = 10
BATCH_SIZE = 64
BUFFER_SIZE = 1000

## main function

def main():
    dataset_loader = DatasetLoader(
        path='D:\Local UNSPLASH Dataset Full',
        feature_lod=5,
        label_lod=3,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        ds_batch_size=1000,
        ds_num_batches=100
    )

    training_data, _ = dataset_loader.load_supervised_dataset(num_images=1000)

    img = cv2.imread('C:\\Users\\lukas\\source\\repos\\comparison-of-SR-methods\\images\\Unsplash_Lite_01_32.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255

    tensor = tf.convert_to_tensor(img, dtype=tf.dtypes.float32)
    sample_images = tf.expand_dims(tensor, axis=0)

    pipeline = Pipeline(
        framework=presets.build_SRDemo(),
        epochs = EPOCHS,
        training_data=training_data,
        sample_images=sample_images
    )

    pipeline.train()


## main function call ##

if __name__ == '__main__':
    main()