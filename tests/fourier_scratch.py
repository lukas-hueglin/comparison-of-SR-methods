### fourier_srcatch.py ###
import os
from PIL import Image
from matplotlib import pyplot as plt

# images folder in ignored by git
scriptDir = os.path.dirname(__file__)
impath = os.path.join(scriptDir, '../images/Unsplash_Lite_01.jpg')

# open image
image = Image.open(impath)

# plot image
plt.imshow(image)
plt.show()