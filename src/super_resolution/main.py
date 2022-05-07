### main.py ###

import tensorflow as tf

from utils import DatasetLoader, SampleLoader, DatasetType
from pipeline import Pipeline

import presets


## global parameters
EPOCHS = 15
BATCH_SIZE = 64
BUFFER_SIZE = 1000

## main function
def main():

    # create dataset loader
    dataset_loader = DatasetLoader(
        path='D:\\Local UNSPLASH Dataset Full',
        feature_lod=5,
        label_lod=3,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        dataset_type=DatasetType.SUPERVISED,
        dataset_size=1000
    )

    # create sample loader
    sample_loader = SampleLoader(
        path='D:\\UNSPLASH Samples',
        lod=5
    )

    # load data
    sample_images = sample_loader.load_samples() 

    # create pipeline
    pipeline = Pipeline(
        framework=presets.build_SRDemo(),
        epochs=EPOCHS,
        epoch_start=0,
        dataset_loader=dataset_loader,
        #path='C:\\Users\\lukas\\source\\repos\\comparison-of-SR-methods\\models\\SRDemo_v.45',
        
        # if you don't have sample images or don't need it just set it None
        sample_images=sample_images
    )

    # load data
    #pipeline.deserialize()

    # train
    pipeline.train()



## main function call ##
if __name__ == '__main__':
    # tensorflow settings
    GPUs = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    main()