### main.py ###

import tensorflow as tf

from utils import DatasetLoader, SampleLoader
from pipeline import Trainer, Validator, Performer

import presets
from utils import TColors


## global parameters
MODE = 'training'


EPOCHS = 40
BATCH_SIZE = 2
BUFFER_SIZE = 1000

## main function
def main():

    # create dataset loader
    dataset_loader = DatasetLoader(
        path='D:\\Local UNSPLASH Dataset Full',
        feature_lod=3,
        label_lod=1,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        dataset_type='unsupervised',
        dataset_size=1000
    )

    # create sample loader
    sample_loader = SampleLoader(
        path='D:\\Local UNSPLASH Samples',
        resolution=128,
        batch_size=BATCH_SIZE
    )

    # choose the mode
    if MODE == 'training':
        # create pipeline
        pipeline = Trainer(
            framework=presets.build_SRGAN(),
            epochs=EPOCHS,
            dataset_loader=dataset_loader,
            # if you don't have sample images or don't need it just set it None
            sample_loader=sample_loader, 
            # load a pretrained framework
            #load_path='SRGAN_v.001'
        )

        # check the variables of pipeline
        pipeline.check()

        # train
        pipeline.train()

    elif MODE == 'validation':
        # create pipeline
        pipeline = Validator(
            dataset_loader=dataset_loader,
            # load a pretrained framework
            load_path='SRGAN_v.001'
        )

        # check the variables of pipeline
        pipeline.check()

        # train
        pipeline.validate()
        
    elif MODE == 'perform':
        # create pipeline
        pipeline = Performer(
            sample_loader=sample_loader,
            # load a pretrained framework
            load_path=None
        )

        # check the variables of pipeline
        pipeline.check()

        # perform superresolution
        pipeline.perform_SR()
        
    else:
        print(TColors.WARNING + 'This Mode does not exist!' + TColors.ENDC)



## main function call ##
if __name__ == '__main__':
    # tensorflow settings
    print(TColors.FAIL + '\nTensorflow' + TColors.NOTE + ' Initialisation:')

    GPUs = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)

    main()