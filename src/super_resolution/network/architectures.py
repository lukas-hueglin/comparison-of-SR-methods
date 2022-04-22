### architectures.py ###

from keras import layers, Sequential

def build_ResNet():
    network = Sequential()
    return network

def build_SRGAN_disc():
    network = Sequential()
    return network

def build_Demo():
    network = Sequential()

    network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    network.add(layers.Conv2D(64, (5, 5), activation='relu'))
    network.add(layers.Conv2D(64, (3, 3), activation='relu'))

    network.add(layers.Conv2DTranspose(32, (3, 3), activation='relu'))
    network.add(layers.Conv2DTranspose(64, (5, 5), activation='relu'))
    network.add(layers.Conv2DTranspose(3, (3, 3), activation='relu'))

    return network