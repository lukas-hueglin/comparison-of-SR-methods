import os
import numpy as np

import cv2
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, RangeSlider


### fourier helper functions ###

# function which masks out the frequencies between min and max
def make_masked(fft, min, max):
    height, width = np.shape(fft)[:2]
    cheight, cwidth = int(height/2), int(width/2)

    mask = np.zeros(np.shape(fft), dtype=complex)
    mask[cheight-max:cheight+max, cwidth-max:cwidth+max] = 1 #low pass filter
    mask[cheight-min:cheight+min, cwidth-min:cwidth+min] = 0 # high pass filter

    return fft * mask

# function which weights the frequencies between min and max
def make_weights(fft, min, max, w0, w1, w2):
    height, width = np.shape(fft)[:2]
    cheight, cwidth = int(height/2), int(width/2)

    weight = np.zeros(np.shape(fft), dtype=complex)
    weight[:, :] = w2
    weight[cheight-max:cheight+max, cwidth-max:cwidth+max] = w1 #low pass filter
    weight[cheight-min:cheight+min, cwidth-min:cwidth+min] = w0 # high pass filter

    return fft * weight

# combines r, g, b channels of shape (y, x) into one image of shape (y, x, 3)
def combine_channels(r, g, b):
    img = np.zeros(np.append(np.shape(r), 3), dtype=complex)

    for y in range(np.shape(r)[0]):
        for x in range(np.shape(r)[1]):
            img[y, x, 0] = r[y, x]
            img[y, x, 1] = g[y, x]
            img[y, x, 2] = b[y, x]
    return img

# seperates an of shape (y, x, 3) into r, g, b channels of shape (y, x)
def seperate_channels(img):
    r = np.zeros(np.shape(img)[:2], dtype=complex)
    g = np.zeros(np.shape(img)[:2], dtype=complex)
    b = np.zeros(np.shape(img)[:2], dtype=complex)

    for y in range(np.shape(img)[0]):
        for x in range(np.shape(img)[1]):
            r[y, x] = img[x, y, 0]
            g[y, x] = img[x, y, 1]
            b[y, x] = img[x, y, 2]
    return (r, g, b)

# performs a fft on all channels
def img_fft(img):
    (red, green, blue) = seperate_channels(img)

    fft_r = np.fft.fftshift(np.fft.fft2(red))
    fft_g = np.fft.fftshift(np.fft.fft2(green))
    fft_b = np.fft.fftshift(np.fft.fft2(blue))

    fft = combine_channels(fft_r, fft_g, fft_b)

    return fft

# performs a ifft on all channels
def img_ifft(fft):
    (fft_r, fft_g, fft_b) = seperate_channels(fft)

    ifft_r = np.fft.ifft2(np.fft.ifftshift(fft_r))
    ifft_g = np.fft.ifft2(np.fft.ifftshift(fft_g))
    ifft_b = np.fft.ifft2(np.fft.ifftshift(fft_b))

    img = abs(combine_channels(ifft_r, ifft_g, ifft_b)) / 255
    return img


def bicubic(img):
    size = (img.shape[0]*2, img.shape[1]*2)
    return cv2.resize(img, dsize=size, interpolation=cv2.INTER_CUBIC)

# (with help from: https://datahacker.rs/004-how-to-smooth-and-sharpen-an-image-in-opencv/)
def sharpen(img, alpha):
    filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img,-1,filter)
    return cv2.addWeighted(img, 1-alpha, sharpened, alpha, 0.0)


def fourier(img, f1, f2, w0, w1, w2):
    fourier = img_fft(img)

    low_fourier = make_masked(fourier, 0, f1)
    mid_fourier = make_masked(fourier, f1, f2)
    high_fourier = make_masked(fourier, f2, 256)
    weighted_fourier = make_weights(fourier, f1, f2, w0, w1, w2)

    low_img = img_ifft(low_fourier)
    mid_img = img_ifft(mid_fourier)
    high_img = img_ifft(high_fourier)
    output = img_ifft(weighted_fourier)

    return (low_img, mid_img, high_img), output

def main():
    # images folder in ignored by git
    scriptDir = os.path.dirname(__file__)
    impath = os.path.join(scriptDir, '../images/Unsplash_Lite_04.jpg')
    #impath = 'D:\\Super Resolution Samples\\LOD_3\\image_026.jpg'

    # open image
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_sharpen = sharpen(image, 0.2)
    image_bicubic = bicubic(image_sharpen)
    #image_sharpen = sharpen(image_bicubic, 0.5)
    images_fourier, image_output = fourier(image_bicubic, 3, 18, 1, 1, 1)

    fig = plt.figure(constrained_layout=True)
    #plt.subplots_adjust(left=0.1, bottom=0.07, right=0.9, top=0.93, wspace=0.1, hspace=0.2)
    gs = fig.add_gridspec(4, 5)

    ax_original = fig.add_subplot(gs[1:3, 0:2])
    ax_bicubic = fig.add_subplot(gs[2:4, 3:5])
    #ax_sharpen = fig.add_subplot(gs[1, 2])
    ax_fourier = (fig.add_subplot(gs[0, 2]), fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 2]))
    ax_output = fig.add_subplot(gs[0:2, 3:5])

    ax_original.set(title='Original')
    ax_bicubic.set(title='Bicubic')
    #ax_sharpen.set(title='Sharpened')
    ax_fourier[0].set(title='Fourier LF')
    ax_fourier[1].set(title='Fourier MF')
    ax_fourier[2].set(title='Fourier HF')
    ax_output.set(title='Output')

    # sliders to control the cut height and the frequency range
    # (with documentation from: https://matplotlib.org/stable/gallery/widgets/slider_demo.html)

    axrange = plt.axes([0.1, 0.015, 0.5, 0.02])
    range_slider = RangeSlider(
        ax=axrange,
        label='Fourier range',
        valmin=0,
        valmax=256,
        valinit=(3, 18),
        orientation='horizontal'
    )

    # controls the weights
    axlow = plt.axes([0.4, 0.8, 0.02, 0.15])
    low_slider = Slider(
        ax=axlow,
        label='Low Weight',
        valmin=0.75,
        valmax=1.25,
        valinit=1,
        orientation='vertical'
    )
    axmid = plt.axes([0.4, 0.54, 0.02, 0.15])
    mid_slider = Slider(
        ax=axmid,
        label='Mid Weight',
        valmin=0.5,
        valmax=1.5,
        valinit=1,
        orientation='vertical'
    )
    axhigh = plt.axes([0.4, 0.3, 0.02, 0.15])
    high_slider = Slider(
        ax=axhigh,
        label='High Weight',
        valmin=0,
        valmax=2,
        valinit=1,
        orientation='vertical'
    )
    axsharp = plt.axes([0.1, 0.1, 0.15, 0.02])
    sharpen_slider = Slider(
        ax=axsharp,
        label='Sharpen alpha',
        valmin=0,
        valmax=1,
        valinit=0.2,
        orientation='horizontal'
    )

    def update_fourier(val):
        ax_fourier[0].clear(), ax_fourier[1].clear(), ax_fourier[2].clear()

        image_sharpen = sharpen(image, sharpen_slider.val)
        image_bicubic = bicubic(image_sharpen)
        images_fourier, image_output = fourier(image_bicubic, int(range_slider.val[0]), int(range_slider.val[1]), low_slider.val, mid_slider.val, high_slider.val)

        ax_bicubic.imshow(image_bicubic)
        ax_bicubic.set(title='Bicubic')
        
        ax_fourier[0].imshow(images_fourier[0])
        ax_fourier[1].imshow(images_fourier[1])
        ax_fourier[2].imshow(images_fourier[2])

        ax_fourier[0].set(title='Fourier LF')
        ax_fourier[1].set(title='Fourier MF')
        ax_fourier[2].set(title='Fourier HF')

        ax_output.imshow(image_output)
        ax_output.set(title='Output')



    def reset():
        ax_original.imshow(image)
        ax_bicubic.imshow(image_bicubic)
        #ax_sharpen.imshow(image_sharpen)
        ax_fourier[0].imshow(images_fourier[0])
        ax_fourier[1].imshow(images_fourier[1])
        ax_fourier[2].imshow(images_fourier[2])
        ax_output.imshow(image_output)

    # connect the update functions to the sliders
    #sharpen_slider.on_changed(update_sharpen)

    range_slider.on_changed(update_fourier)

    low_slider.on_changed(update_fourier)
    mid_slider.on_changed(update_fourier)
    high_slider.on_changed(update_fourier)
    sharpen_slider.on_changed(update_fourier)

    # plot
    reset()
    plt.show()


## main function call ##
if __name__ == '__main__':
    main()