from msilib.schema import RadioButton
import os
import numpy as np

from functools import partial
from multiprocessing import Pool

import cv2
from matplotlib import pyplot as plt

# Global params
SLICES = 12
RADIUS = 4
UPSCALING_FACTOR = 4

# helper functions

def get_slice_from_points(xr, yr, xp, yp):
    delta_x = xp - xr
    delta_y = yr - yp

    if delta_x == 0:
        angle = np.pi/2
    else:
        angle = abs(np.arctan(delta_y / delta_x))
    if delta_x <= 0 and delta_y > 0:
        angle = np.pi - angle
    elif delta_x <= 0 and delta_y < 0:
        angle = angle + np.pi
    elif delta_x >= 0 and delta_y < 0:
        angle = 2*np.pi - angle

    return int(np.floor(SLICES * angle / (2*np.pi))) + 1

def encode_row(y, img):
    row = []
    for x in range(len(img[y])):
        channels = []
        for c in range(3):
            slices = []
            slices.append(img[y, x, c])
            for a in range(SLICES):
                values = []
                for i in range(RADIUS):
                    new_x = int(np.round(x+np.cos(2*a*np.pi/SLICES)*i))
                    new_y = int(np.round(y+np.sin(2*a*np.pi/SLICES)*i))

                    if new_x >= len(img[y]) or new_y >= len(img) or new_x < 0 or new_y < 0:
                        values.append(0)
                    else:
                        values.append(img[new_y, new_x, c])
                slices.append(np.mean(values))
            channels.append(slices)
        row.append(channels)
    return row

def decode_row(y, img):
    encoded_y = int(y / UPSCALING_FACTOR)
    row = []
    for x in range(len(img[encoded_y]) * UPSCALING_FACTOR):
        encoded_x = int(x / UPSCALING_FACTOR)
        channels = []
        for c in range(3):
            sum = 0
            sum_of_weights = 0
            for py in range(encoded_y-RADIUS, encoded_y+RADIUS+1):
                for px in range(encoded_x-RADIUS, encoded_x+RADIUS+1):
                    slice = get_slice_from_points(encoded_x, encoded_y, px, py)
                    distance = np.sqrt(np.square(encoded_x-px) + np.square(encoded_y-py))
                    weight = 1

                    if distance > RADIUS:
                        weight = 0
                    if px >= len(img[encoded_y]) or py >= len(img) or px < 0 or py < 0:
                        weight = 0
                    else:
                        sum += weight * img[py][px][c][slice]
                        sum_of_weights += weight

            if sum_of_weights == 0:
                channels.append(0)
            else:
                channels.append(int(np.round(sum / sum_of_weights)))
        row.append(channels)
    return row


def main():
    # images folder in ignored by git
    scriptDir = os.path.dirname(__file__)
    impath = os.path.join(scriptDir, '../images/Unsplash_Lite_04.jpg')

    # open image
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # encode the image before upscaling
    encoded_image = []
    with Pool(processes=10) as pool:
        func = partial(encode_row, img=image)
        encoded_image = pool.map(func, range(len(image)))

    print('encoding finished!')

    # decode the image
    decoded_image = []
    with Pool(processes=10) as pool:
        func = partial(decode_row, img=encoded_image)
        decoded_image = pool.map(func, range(len(image) * UPSCALING_FACTOR))

    print('decoding finished!')

    fig, (img1, img2, img3) = plt.subplots(1, 3)

    img1.imshow(image)
    img2.imshow(cv2.resize(image, (np.shape(image)[0] * UPSCALING_FACTOR, np.shape(image)[1] * UPSCALING_FACTOR), interpolation=cv2.INTER_CUBIC))
    img3.imshow(decoded_image)

    # set titles
    img1.title.set_text('Original')
    img2.title.set_text('Bicubic')
    img3.title.set_text('Generated')

    plt.show()
    


## main function call ##
if __name__ == '__main__':
    main()