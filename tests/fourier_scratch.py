### fourier_scratch.py ###
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


### plot helper functions ###

# plots the rgb values of the chosen row of the original image
def dynamic_plot(x, y, ax, cut):
    ax.plot(x, y[cut, :, 2], 'b', x, y[cut, :, 1], 'g', x, y[cut, :, 0], 'r')
    ax.set(xlabel='Pixels', ylabel='Brightness')

# plots the fft of the chosen row of the original image
def dynamic_plot_fft2(fft, freq, ax, cut):
    ax.plot(freq, abs(fft[cut, 0:256, 2]/256), 'b', freq, abs(fft[cut, 0:256, 1]/256), 'g', freq, abs(fft[cut, 0:256, 0]/256), 'r')
    ax.set(xlabel='Frequency (1 Hz corresponds to the image width)', ylabel='Amplitude')

# plots the original image with a horizontal line for the cut height
def dynamic_imshow(img, ax, cut):
    ax.imshow(img)
    ax.axhline(y=cut, color='red')
    ax.set(title='Original image')

# plots the inverted fft with an applied frequency filter
def dynamic_imshow_ifft(fft, ax, min, max):
    ifft = img_ifft(make_masked(fft, min, max))
    ax.imshow(ifft)
    ax.set(title='Image with frequency filter')


# images folder in ignored by git
scriptDir = os.path.dirname(__file__)
impath = os.path.join(scriptDir, '../images/Unsplash_Lite_03.jpg')

# open image
image = cv2.imread(impath)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# fast fourier transform
# (with documentation from: https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft
# and from: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_transforms/py_fourier_transform/py_fourier_transform.html)

fft = img_fft(image)
freq = np.linspace(1, 256, num=256)
width = np.linspace(1, 512, num=512)

# prepare plots
# (with documentation from: https://matplotlib.org/3.1.0/tutorials/intermediate/gridspec.html)
fig = plt.figure(constrained_layout=True)
plt.subplots_adjust(left=0.1, bottom=0.07, right=0.9, top=0.93, wspace=0.1, hspace=0.2)
gs = fig.add_gridspec(3, 2)

axs = [fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
        fig.add_subplot(gs[2, :])]

# sliders to control the cut height and the frequency range
# (with documentation from: https://matplotlib.org/stable/gallery/widgets/slider_demo.html)

# controls which row of the image will be shown in the rgb value graph and the fft graph
axcut = plt.axes([0.185, 0.675, 0.01, 0.255])
cut_slider = Slider(
    ax=axcut,
    label='Cut height',
    valmin=1,
    valmax=512,
    valinit=256,
    orientation='vertical'
)

# controls which frequencies won't be masked out
axifft = plt.axes([0.1, 0.015, 0.8, 0.02])
ifft_slider = RangeSlider(
    ax=axifft,
    label='Frequency Range',
    valmin=0,
    valmax=256,
    valinit=(0, 256)
)

# update functions for the sliders
def update_cut(val):
    axs[0].clear(), axs[2].clear(), axs[3].clear()

    dynamic_imshow(image, axs[0], int(val))
    dynamic_plot(width, image, axs[2], int(val))
    dynamic_plot_fft2(fft, freq, axs[3], int(val))

def update_ifft(val):
    axs[1].clear()
    dynamic_imshow_ifft(fft, axs[1], int(val[0]), int(val[1]))

def reset():
    for ax in axs:
        ax.clear()
    dynamic_imshow(image, axs[0], 256)
    dynamic_imshow_ifft(fft, axs[1], 0, 256)
    dynamic_plot(width, image, axs[2], 256)
    dynamic_plot_fft2(fft, freq, axs[3], 256)

# connect the update functions to the sliders
cut_slider.on_changed(update_cut)
ifft_slider.on_changed(update_ifft)

# plot
reset()
plt.show()