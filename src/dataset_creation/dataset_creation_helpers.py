### dataset_creation_helpers.py ###
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import datetime

from image_downloading import download_images_multiproc

# colors for terminal output
# (from: https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal)
class TColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# creates a new directory if it does not exist
def make_dir(path):
    dirs = path.split(('\\'))
    joined_path = dirs[0]

    for dir in dirs[1:]:
        joined_path = os.path.join(joined_path, dir)
        if os.path.exists(joined_path) is False:
            os.mkdir(joined_path)

def resize_images(imgs, size):
    output_imgs = []

    for img in imgs:
        output_imgs.append(cv2.resize(img, (size, size)))

    return np.array(output_imgs)

def save_images(path, imgs, sizes, start_index):
    for s in range(len(sizes)):
        dir = int(np.floor(start_index / 1e4))
        sub_dir = int(np.floor(start_index / 1e3))
        cropped_imgs = resize_images(imgs, sizes[s])

        for i in range(start_index, np.shape(imgs)[0] + start_index):
            if i % 1e4 == 0 and i!= 0:
                dir += 1
                sub_dir = 0
            elif i % 1e3 == 0 and i!= 0:
                sub_dir += 1

            dir_path = os.path.join(path, 'LOD_' + str(s), str(dir), str(sub_dir))
            img_path = os.path.join(dir_path, 'image_' + str(i) + '.jpg')

            make_dir(dir_path)

            cv2.imwrite(os.path.join(img_path, img_path), cropped_imgs[i - start_index])

class DatasetCreator:
    def __init__(self, path_in, path_out, sizes=(256, 64), dataset_size=2e4,
                download_ratio=1, dataset_type='Lite', author_name='unknown'):

        self.path_in = path_in
        self.path_out = path_out

        self.sizes = np.sort(sizes)[::-1]

        self.dataset_size = dataset_size
        self.download_ratio = download_ratio

        self.dataframe = pd.DataFrame()

        self.dataset_type = dataset_type
        self.author_name = author_name

    # loads 'photos.tsv000' from self.path_in into a dataframe
    def load_dataframe(self):
        self.dataframe = pd.read_csv(self.path_in+'\\photos.tsv000', sep='\t', header=0)

    # loads all the image urls into self.images
    # (for the future: here could be somekind of search algorithm for differing different types implemented)
    def search_images(self):
        urls = np.array(self.dataframe.loc[:, 'photo_image_url'])

        # prints a warning if user requests to many images
        if len(urls) < self.dataset_size:
            print(TColors.WARNING + 'Warning: The Unsplash Dataset contains just ' + str(len(urls)) + ' images!' + TColors.ENDC)
        else:
            urls = urls[0:int(self.dataset_size)]

        return urls

    # downloads all the images from self.images and saves them in self.path_out
    # this function is still messy and should be cleaned up!
    def download_images(self, urls, batch_size, num_proc):
        num_batches = int(len(urls) / batch_size)
        split_urls = np.split(urls, num_batches)

        for batch in tqdm(range(len(split_urls))):
            images = download_images_multiproc(split_urls[batch], np.amax(self.sizes), num_proc)

            data_path = os.path.join(self.path_out, 'data')
            save_images(data_path, images, self.sizes, batch*batch_size)

    # saves self.images in a .csv file
    def save_cache(self, urls):
        df = pd.DataFrame(urls)
        df.to_csv(self.path_out+'\\cache.csv')

    # creates a README.md file with information about the locally downloaded dataset
    def make_README(self):
        # create content in markdown
        text = '# Local download of the Unsplash Dataset\n\n'

        text += '![](https://drive.google.com/uc?export=view&id=1DZz--VgBa2dxYne4ATCbTUdNGY4W79S4)\n\n'

        text += '## 1 - Unsplash Dataset Access\n\n'

        text += '*Note: Request access at [Unsplash Dataset](https://unsplash.com/data)*\n\n'

        text += 'Dataset type: ' + self.dataset_type + '<br> \n'
        text += 'Name: ' + self.author_name + '\n\n'
                
        text+= 'Access date: ' + str(datetime.date.today()) + '\n\n'

        text += '## 2 - Dataset Parameters\n\n'

        text += 'Downloaded images: ' + str(int(self.dataset_size*self.download_ratio)) + ' (' + str(len(self.images) - int(self.dataset_size*self.download_ratio)) + ' remaining)' + '<br> \n'
        text += 'Images in `cache.csv`: ' + str(len(self.images))+ '\n\n'
                
        text += '## 2 - Image Resolutions\n\n'

        text += '| Level of Detail             | Image Resolution |\n'
        text += '|-----------------------------|------------------|'

        for s in range(len(self.sizes)):
            text += '\n| LOD_'+str(s) + '                       | ('
            text += str(self.sizes[s]) + ' x ' + str(self.sizes[s]) + ')    |'

        # create and save README.md file
        file = open(self.path_out+'\\README.md', 'a')
        file.write(text)
        file.close()