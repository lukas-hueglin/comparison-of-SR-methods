import os
import numpy as np

import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw


def generate_sample():
    sample = abs(np.random.normal(size=(6, 6))) * 255

    '''
    sample = np.zeros(shape=(6, 6))

    for i in range(6):
        for j in range(6):
            sample[i, j] = (i / 10 + j / 10) * 255'''

    return sample

def scale_up(img):
    upscale = Image.fromarray(img).resize((img.shape[0]*2, img.shape[1]*2))
    draw = ImageDraw.Draw(upscale) 

    for y in range(img.shape[0]-1):
        for x in range(img.shape[1]-1):
            if abs(img[y, x]-img[y, x+1]) <= abs(img[y, x]-img[y+1, x+1]):
                if abs(img[y, x]-img[y, x+1]) <= abs(img[y, x]-img[y+1, x]):
                    if abs(img[y, x]-img[y, x+1]) <= abs(img[y, x]-img[y+1, x-1]):
                        i = x+1
                        j = y
                    else:
                        i = x -1
                        j = y + 1
                elif abs(img[y, x]-img[y+1, x]) <= abs(img[y, x]-img[y+1, x-1]):
                    i = x
                    j = y+1
                else:
                    i = x -1
                    j = y + 1
            elif abs(img[y, x]-img[y+1, x+1]) <= abs(img[y, x]-img[y+1, x]):
                if abs(img[y, x]-img[y+1, x+1]) <= abs(img[y, x]-img[y+1, x-1]):
                    i = x+1
                    j = y+1
                else:
                    i = x -1
                    j = y + 1 
            elif abs(img[y, x]-img[y+1, x]) <= abs(img[y, x]-img[y+1, x-1]):
                i = x
                j = y+1
            else:
                i = x -1
                j = y + 1
            
            draw.line((2*x, 2*y, 2*i, 2*j), fill=int(img[y, x]), width=1)

            '''
            fig, (small, big) = plt.subplots(1, 2)
            small.imshow(img, cmap='gray')
            small.scatter((x, i), (y, j), color=('red', 'blue'))
            big.imshow(upscale, cmap='gray')
            big.scatter((2*x, 2*i), (2*y, 2*j), color=('red', 'blue'))
            plt.show()'''

    return upscale
   

def main():
    # images folder in ignored by git
    scriptDir = os.path.dirname(__file__)
    impath = os.path.join(scriptDir, '../images/img_04.jpg')

    # open image
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    r = image[:,:,0]
    g = image[:,:,1]
    b = image[:,:,2]

    upscale_1_r = np.array(scale_up(r))
    upscale_1_g = np.array(scale_up(g))
    upscale_1_b = np.array(scale_up(b))

    upscale_2_r = np.array(scale_up(upscale_1_r))
    upscale_2_g = np.array(scale_up(upscale_1_g))
    upscale_2_b = np.array(scale_up(upscale_1_b))

    upscale = cv2.merge((upscale_2_r, upscale_2_g, upscale_2_b))

    fig, (small, big) = plt.subplots(1, 2)
    small.imshow(image)
    big.imshow(upscale)
    plt.show()


## main function call ##
if __name__ == '__main__':
    main()