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


def fourier(img, f1, f2):
    fourier = img_fft(img)

    low_fourier = make_masked(fourier, 0, f1)
    mid_fourier = make_masked(fourier, f1, f2)
    high_fourier = make_masked(fourier, f2, 256)

    low_img = img_ifft(low_fourier)
    mid_img = img_ifft(mid_fourier)
    high_img = img_ifft(high_fourier)

    return (low_img, mid_img, high_img)

def noise(img, s1, s2, s3):
    size = (img.shape[0], img.shape[1])
    low = np.random.normal(1, s1, size=size)
    mid = np.random.normal(1, s2, size=size)
    high = np.random.normal(1, s3, size=size)

    img_low = cv2.merge((low, low, low))
    img_mid = cv2.merge((mid, mid, mid))
    img_high = cv2.merge((high, high, high))

    return (img_low * img/255, img_mid * img/255, img_high * img/255)

def additive_noise(noise, fourier):
    return (noise[0] * fourier[0], noise[1] * fourier[1], noise[2] * fourier[2])

def output(img, details, fourier):
    output = (img/255 * np.ones_like(fourier[0]) - (fourier[0] + fourier[1] + fourier[2])/3) + ((details[0] * fourier[0]) + (details[1] * fourier[1]) + (details[2] * fourier[2]))/3
    return output

def main():
    # images folder in ignored by git
    scriptDir = os.path.dirname(__file__)
    impath = os.path.join(scriptDir, '../images/Unsplash_Lite_05.jpg')
    impath = 'D:\\Super Resolution Samples\\LOD_3\\image_037.jpg'

    # open image
    image = cv2.imread(impath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image_bicubic = bicubic(image)
    image_sharpen = sharpen(image_bicubic, 0.5)
    images_fourier = fourier(image_bicubic, 3, 18)
    image_noise = noise(image_bicubic, 0.1, 0.15, 0.4)
    image_additive_noise = additive_noise(image_noise, images_fourier)
    image_output = output(image_bicubic, image_noise, images_fourier)

    fig = plt.figure(constrained_layout=True)
    #plt.subplots_adjust(left=0.1, bottom=0.07, right=0.9, top=0.93, wspace=0.1, hspace=0.2)
    gs = fig.add_gridspec(3, 8)

    ax_original = fig.add_subplot(gs[0:2, 0:2])
    ax_bicubic = fig.add_subplot(gs[0, 2])
    ax_sharpen = fig.add_subplot(gs[1, 2])
    ax_fourier = (fig.add_subplot(gs[0, 3]), fig.add_subplot(gs[1, 3]), fig.add_subplot(gs[2, 3]))
    ax_noise = (fig.add_subplot(gs[0, 4]), fig.add_subplot(gs[1, 4]), fig.add_subplot(gs[2, 4]))
    ax_additive_noise = (fig.add_subplot(gs[0, 5]), fig.add_subplot(gs[1, 5]), fig.add_subplot(gs[2, 5]))
    ax_output = fig.add_subplot(gs[0:2, 6:8])

    ax_original.set(title='Original')
    ax_bicubic.set(title='Bicubic')
    ax_sharpen.set(title='Sharpened')
    ax_fourier[0].set(title='Fourier LF')
    ax_fourier[1].set(title='Fourier MF')
    ax_fourier[2].set(title='Fourier HF')
    ax_noise[0].set(title='Noise LF')
    ax_noise[1].set(title='Noise MF')
    ax_noise[2].set(title='Noise HF')
    ax_additive_noise[0].set(title='Details LF')
    ax_additive_noise[1].set(title='Details MF')
    ax_additive_noise[2].set(title='Details HF')
    ax_output.set(title='Output')

    # sliders to control the cut height and the frequency range
    # (with documentation from: https://matplotlib.org/stable/gallery/widgets/slider_demo.html)

    # controls which frequencies won't be masked out
    axsharpen = plt.axes([0.1, 0.07, 0.02, 0.33])
    sharpen_slider = Slider(
        ax=axsharpen,
        label='Sharpen',
        valmin=0,
        valmax=1,
        valinit=0.5,
        orientation='vertical'
    )

    axrange = plt.axes([0.1, 0.015, 0.8, 0.02])
    range_slider = RangeSlider(
        ax=axrange,
        label='Fourier range',
        valmin=0,
        valmax=256,
        valinit=(3, 18),
        orientation='horizontal'
    )

    # controls the random noise
    axlow = plt.axes([0.8, 0.07, 0.02, 0.33])
    low_slider = Slider(
        ax=axlow,
        label='Low Sigma',
        valmin=0,
        valmax=2,
        valinit=0.1,
        orientation='vertical'
    )
    axmid = plt.axes([0.875, 0.07, 0.02, 0.33])
    mid_slider = Slider(
        ax=axmid,
        label='Mid Sigma',
        valmin=0,
        valmax=2,
        valinit=0.15,
        orientation='vertical'
    )
    axhigh = plt.axes([0.95, 0.07, 0.02, 0.33])
    high_slider = Slider(
        ax=axhigh,
        label='High Sigma',
        valmin=0,
        valmax=2,
        valinit=0.4,
        orientation='vertical'
    )

    # update functions for the sliders
    def update_sharpen(val):
        ax_sharpen.clear()
        image_sharpen = sharpen(image_bicubic, val)

        ax_sharpen.imshow(image_sharpen)
        ax_sharpen.set(title='Sharpened')

        update_range(range_slider.val)

    def update_range(val):
        ax_fourier[0].clear(), ax_fourier[1].clear(), ax_fourier[2].clear()

        images_fourier = fourier(image_bicubic, int(val[0]), int(val[1]))
        ax_fourier[0].imshow(images_fourier[0])
        ax_fourier[1].imshow(images_fourier[1])
        ax_fourier[2].imshow(images_fourier[2])

        ax_fourier[0].set(title='Fourier LF')
        ax_fourier[1].set(title='Fourier MF')
        ax_fourier[2].set(title='Fourier HF')

        update_low(low_slider.val)
        update_mid(mid_slider.val)
        update_high(high_slider.val)

    def update_low(val):
        ax_noise[0].clear(), ax_additive_noise[0].clear(), ax_output.clear()
        image_noise = noise(image_bicubic, val, mid_slider.val, high_slider.val)
        image_additive_noise = additive_noise(image_noise, images_fourier)
        image_output = output(image_bicubic, image_noise, images_fourier)

        ax_noise[0].imshow(image_noise[0])
        ax_additive_noise[0].imshow(image_additive_noise[0])
        ax_output.imshow(image_output)

        ax_noise[0].set(title='Noise LF')
        ax_additive_noise[0].set(title='Details LF')
        ax_output.set(title='Output')
        
    def update_mid(val):
        ax_noise[1].clear(), ax_additive_noise[1].clear(), ax_output.clear()
        image_noise = noise(image_bicubic, low_slider.val, val, high_slider.val)
        image_additive_noise = additive_noise(image_noise, images_fourier)
        image_output = output(image_bicubic, image_noise, images_fourier)

        ax_noise[1].imshow(image_noise[1])
        ax_additive_noise[1].imshow(image_additive_noise[1])
        ax_output.imshow(image_output)

        ax_noise[1].set(title='Noise MF')
        ax_additive_noise[1].set(title='Details MF')
        ax_output.set(title='Output')

    def update_high(val):
        ax_noise[2].clear(), ax_additive_noise[2].clear(), ax_output.clear()
        image_noise = noise(image_bicubic, low_slider.val, mid_slider.val, val)
        image_additive_noise = additive_noise(image_noise, images_fourier)
        image_output = output(image_bicubic, image_noise, images_fourier)

        ax_noise[2].imshow(image_noise[2])
        ax_additive_noise[2].imshow(image_additive_noise[2])
        ax_output.imshow(image_output)

        ax_noise[2].set(title='Noise HF')
        ax_additive_noise[2].set(title='Details HF')
        ax_output.set(title='Output')


    def reset():
        ax_original.imshow(image)
        ax_bicubic.imshow(image_bicubic)
        ax_sharpen.imshow(image_sharpen)
        ax_fourier[0].imshow(images_fourier[0])
        ax_fourier[1].imshow(images_fourier[1])
        ax_fourier[2].imshow(images_fourier[2])
        ax_noise[0].imshow(image_noise[0])
        ax_noise[1].imshow(image_noise[1])
        ax_noise[2].imshow(image_noise[2])
        ax_additive_noise[0].imshow(image_additive_noise[0])
        ax_additive_noise[1].imshow(image_additive_noise[1])
        ax_additive_noise[2].imshow(image_additive_noise[2])
        ax_output.imshow(image_output)

    # connect the update functions to the sliders
    sharpen_slider.on_changed(update_sharpen)

    range_slider.on_changed(update_range)

    low_slider.on_changed(update_low)
    mid_slider.on_changed(update_mid)
    high_slider.on_changed(update_high)

    # plot
    reset()
    plt.show()


## main function call ##
if __name__ == '__main__':
    main()