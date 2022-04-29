### architectures.py ###
# This module includes the build functions, which return a specific tensorflow
# model architecture. Every function with the make_ prefix will return such a network.
##

from keras import layers, Sequential


# A placeholder function for the SRResNet architecture
def make_SRResNet(input_res):
    network = Sequential()
    return network, 'SRResNet'

# A placeholder function for the SRGan discriminator
def make_SRGAN_disc(input_res):
    network = Sequential()
    return network, 'SRGAN_disc'

# This network is just for test purposes and isn't build for achieving great accuracies
def make_Demo(input_res):
    network = Sequential()

    network.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(input_res, input_res, 3)))
    network.add(layers.Conv2D(64, (5, 5), activation='relu'))
    network.add(layers.Conv2D(64, (3, 3), activation='relu'))

    network.add(layers.Conv2DTranspose(32, (3, 3), activation='relu'))
    network.add(layers.Conv2DTranspose(64, (5, 5), activation='relu'))
    network.add(layers.Conv2DTranspose(3, (3, 3), activation='relu'))

    return network, 'Demo'