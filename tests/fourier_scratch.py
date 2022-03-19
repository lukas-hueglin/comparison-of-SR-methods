### fourier_srcatch.py ###
import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

def make_mask(width, height, min, max):
    img = np.zeros((width, height))
    for y in range(width):
        for x in range(height):
            d = np.sqrt(np.square(x)+np.square(y))
            if d >= min and d <= max:
                img[y, x] = 1
    return img

def calc_high_freq(fft):
    bias = np.amax(abs(fft)) / 1e3
    hf_density = 0
    for graph in fft:
        for i in range(len(graph)-1, 0, -1):
            if(abs(graph[i])) > bias:
                hf_density += i
                break
    return hf_density


# images folder in ignored by git
scriptDir = os.path.dirname(__file__)
impath = os.path.join(scriptDir, '../images/Unsplash_Lite_02.jpg')

# open image
image = Image.open(impath)
channels = image.split()

# fast fourier transform
# (with documentation from: https://numpy.org/doc/stable/reference/generated/numpy.fft.fft.html#numpy.fft.fft
width = np.arange(512)
amplitude = np.array(channels[0])
fft = np.fft.fft2(amplitude)
freq = np.fft.fftfreq(width.shape[-1])

# mask frequencies
mask = make_mask(512, 512, 0, 10) # the min and max values are not frequencies!!
masked_fft = fft*mask

# inverse fast fourier transform
ifft = np.fft.ifft2(masked_fft)

# plot image
fig, axs = plt.subplots(2)
axs[0].imshow(amplitude, cmap='gray')
axs[1].imshow(abs(ifft), cmap='gray')
plt.show()

# calcultate a measurement for high frequency details
hf_density = calc_high_freq(fft)
print(hf_density)