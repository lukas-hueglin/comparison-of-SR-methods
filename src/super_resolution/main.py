### main.py ###

import tensorflow as tf

from utils import DatasetLoader, SampleLoader, DatasetType
from pipeline import Trainer, Validator, Performer

import presets
from utils import TColors


## global parameters
EPOCHS = 5
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
        lod=5,
        batch_size=BATCH_SIZE
    )

    # create pipeline
    pipeline = Trainer(
        framework=presets.build_SRDemo(),
        epochs=EPOCHS,
        dataset_loader=dataset_loader,
        
        # if you don't have sample images or don't need it just set it None
        sample_loader=sample_loader
    )

    pipeline.check()

    #pipeline.load_framework('SRDemo_v.004')

    # train
    pipeline.train()



## main function call ##
if __name__ == '__main__':
    # tensorflow settings
    print(TColors.FAIL + '\nTensorflow' + TColors.NOTE + ' Initialisation:')

    GPUs = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    main()