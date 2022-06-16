import os
import numpy as np

import cv2
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw

def draw_line(img, p1, p2, c1, c2):
    mean_c = ((c1 + c2) / 2)

    delta_x = p2[0] - p1[0]
    delta_y = p2[1] - p1[1]

    if delta_x != 0 and delta_y != 0:
        if delta_y < 0:
            img[p2[1]-1, p2[0]] = mean_c
            img[p2[1], p2[0]-1] = mean_c
        else:
            img[p2[1]-1, p2[0]] = mean_c
            img[p2[1], p2[0]+1] = mean_c



def generate_sample():
    sample = abs(np.random.normal(size=(6, 6))) * 255

    '''
    sample = np.zeros(shape=(6, 6))

    for i in range(6):
        for j in range(6):
            sample[i, j] = (i / 10 + j / 10) * 255'''

    return sample

def color_dif(c1, c2):
    return np.sqrt(np.square(c1[0] - c2[0]) + np.square(c1[1] - c2[1]) + np.square(c1[2] - c2[2]))

def scale_up(img):
    upscale = np.array(Image.fromarray(img).resize((img.shape[0]*2, img.shape[1]*2), Image.NONE))
    #upscale = Image.new('RGB', (img.shape[0]*2, img.shape[1]*2), (0, 0, 0))
    #upscale = np.zeros(shape=(img.shape[0]*2, img.shape[1]*2, 3))
    #draw = ImageDraw.Draw(upscale) 


    plt.imshow(upscale)
    plt.show()

    for y in range(img.shape[0]-1):
        for x in range(img.shape[1]-1):
            if color_dif(img[y, x], img[y, x+1]) <= color_dif(img[y, x], img[y+1, x+1]):
                if color_dif(img[y, x], img[y, x+1]) <= color_dif(img[y, x], img[y+1, x]):
                    if color_dif(img[y, x], img[y, x+1]) <= color_dif(img[y, x], img[y+1, x-1]):
                        i = x+1
                        j = y
                    else:
                        i = x -1
                        j = y + 1
                elif color_dif(img[y, x], img[y+1, x]) <= color_dif(img[y, x], img[y+1, x-1]):
                    i = x
                    j = y+1
                else:
                    i = x -1
                    j = y + 1
            elif color_dif(img[y, x], img[y+1, x+1]) <= color_dif(img[y, x], img[y+1, x]):
                if color_dif(img[y, x], img[y+1, x+1]) <= color_dif(img[y, x], img[y+1, x-1]):
                    i = x+1
                    j = y+1
                else:
                    i = x -1
                    j = y + 1 
            elif color_dif(img[y, x], img[y+1, x]) <= color_dif(img[y, x], img[y+1, x-1]):
                i = x
                j = y+1
            else:
                i = x -1
                j = y + 1
            
            #draw.line((2*x, 2*y, 2*i, 2*j), fill=(img[y, x, 0], img[y, x, 1], img[y, x, 2]), width=2)
            draw_line(upscale, (2*x, 2*y), (2*i, 2*j), img[y, x], img[j, i])

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

    upscale = scale_up(image)

    #upscale_2_r = np.array(scale_up(upscale_1_r))
    #upscale_2_g = np.array(scale_up(upscale_1_g))
    #upscale_2_b = np.array(scale_up(upscale_1_b))

    fig, (small, bicubic, big) = plt.subplots(1, 3)
    small.imshow(image)
    bicubic.imshow(cv2.resize(image, dsize=(image.shape[0]*2, image.shape[1]*2), interpolation=cv2.INTER_CUBIC))
    big.imshow(upscale)
    plt.show()


## main function call ##
if __name__ == '__main__':
    main()