### main.py ###

from utils import DatasetLoader
from pipeline import Pipeline

import presets

## global parameters

EPOCHS = 10
BATCH_SIZE = 256
BUFFER_SIZE = 1000

## main function

def main():
    dataset_loader = DatasetLoader(
        path='D:\Local UNSPLASH Dataset Full',
        feature_lod=5,
        label_lod=4,
        batch_size=BATCH_SIZE,
        buffer_size=BUFFER_SIZE,
        ds_batch_size=1000,
        ds_num_batches=100
    )

    training_data, _ = dataset_loader.load_supervised_dataset(num_images=10)

    pipeline = Pipeline(
        framework=presets.build_SRDemo(),
        epochs = EPOCHS,
        training_data=training_data,
    )

    pipeline.train()


## main function call ##

if __name__ == '__main__':
    main()