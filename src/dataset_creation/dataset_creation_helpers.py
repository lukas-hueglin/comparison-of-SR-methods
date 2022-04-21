### dataset_creation_helpers.py ###
from ast import comprehension
import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm
import datetime
import h5py

from dataset_creation import image_downloader


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

# resizes an array of images to de desired squared size
def resize_images(imgs, size, interpolation):
    output_imgs = []

    for img in imgs:
        output_imgs.append(cv2.resize(img, (size, size), interpolation=interpolation))

    return np.array(output_imgs)

# saves all LODs of an array of images into a hdf5 file
# (with help from: "GANs mit PyTorch selbst programmieren",
# page: "112", link: https://github.com/makeyourownneuralnetwork/gan/blob/master/10_celeba_download_make_hdf5.ipynb)
# (and with halp from: https://stackoverflow.com/questions/66631284/convert-a-folder-comprising-jpeg-images-to-hdf5/66641176#66641176)
def save_images(path, imgs, sizes, batch_index, interpolation):
    make_dir(path)

    for s in range(len(sizes)):
        # resize image
        cropped_imgs = resize_images(imgs, sizes[s], interpolation)
        # specify the location of the hdf5_file
        hdf5_file = os.path.join(path, 'LOD_' + str(s) + '.hdf5')

        # open or create .hdf5 file
        with h5py.File(hdf5_file, 'a') as hf:
            for i in range(np.shape(cropped_imgs)[0]):
                name = 'BATCH_' + str(batch_index) + '/image_' + str(i)
                try:
                    hf.create_dataset(name, data=cropped_imgs[i], dtype=int, compression='gzip')
                except ValueError:
                    print(TColors.WARNING + name + ' already exists!')



class DatasetCreator:
    def __init__(self, path_in, path_out, sizes=(256, 64), dataset_size=2e4,
                interpolation = cv2.INTER_CUBIC,dataset_type='Lite', author_name='unknown'):

        self.path_in = path_in
        self.path_out = path_out

        self.sizes = np.sort(sizes)[::-1]
        self.interpolation = interpolation

        self.dataset_size = dataset_size

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

    # downloads all the images from urls and saves them in self.path_out
    def download_images(self, urls, batch_size, num_proc, batch_range=(0, -1)):
        num_batches = int(len(urls) / batch_size)

        # change batch_range if there was no custom entry made
        if batch_range[0] > batch_range[1]:
            batch_range = (0, num_batches)

        split_urls = np.split(urls, num_batches)

        # check if the batch_range is out of range
        if num_batches < batch_range[1]:
            print(TColors.WARNING + 'Warning: There are just ' + str(len(split_urls)) + ' batches!' + TColors.ENDC)
        else:
            # download all the images in num_batches steps
            for batch in tqdm(range(batch_range[0], batch_range[1])):
                images = image_downloader.retrieve_images_multiproc(split_urls[batch], np.amax(self.sizes), num_proc)

                data_path = os.path.join(self.path_out, 'data')
                save_images(data_path, images, self.sizes, batch, self.interpolation)

    # saves self.images in a .csv file
    def save_cache(self, urls):
        df = pd.DataFrame(urls)
        df.to_csv(self.path_out+'\\cache.csv')

    def estimate_dataset_size(self):
        space = 0
        a = 2.01462e-6
        b = -5.93564e-5
        c = 1.92251e-3
        d = 5.25465
        e = 2315.02
        for s in self.sizes:
            space += (a * np.square(s) + b * s + c)/(d * s + e)

        print('Estimated dataset size: ' + TColors.BOLD + str(np.round(space * self.dataset_size, 2)) + TColors.ENDC + ' GB')


    # returns the space used by the dataset
    # (it is really slow and can be improved, maybee with: https://www.geeksforgeeks.org/how-to-get-size-of-folder-using-python/)
    def get_used_space(self):
        data_path = os.path.join(self.path_out, 'data')
        size = 0

        for root, dirs, files in os.walk(data_path, topdown=False):
            for f in files:
                size += os.path.getsize(os.path.join(root, f))
        
        return size

    # returns the amount of images
    def get_number_images(self):
        data_path = os.path.join(self.path_out, 'data', 'LOD_0.hdf5')
        number = 0

        with h5py.File(data_path, 'r') as hf:
            names = hf.keys()
            for n in names:
                number += np.shape(hf[n])[0]

        return number

    # converts the images in a folder structure to a .hdf5 file
    def make_hdf5(self, batch_range=(0, -1)):
        # change batch_range if there was no custom entry made
        if batch_range[0] > batch_range[1]:
            batch_range = (0, len(os.listdir(lod0_path)))

        data_path = os.path.join(self.path_out, 'data')
        lod0_path = os.path.join(data_path, 'LOD_0')

        # iterate over all batches
        for batch in tqdm(range(batch_range[0], batch_range[1])):
            batch_path = os.path.join(lod0_path, 'BATCH_'+str(batch))
            imgs = []

            # load all images
            for img in os.listdir(batch_path):
                img = cv2.imread(os.path.join(batch_path, img))[:, :, [2, 1, 0]]
                imgs.append(img)

            # save all images
            save_images(data_path, imgs, self.sizes, batch, self.interpolation)

            

    # creates a README.md file with information about the locally downloaded dataset
    def make_README(self, urls, batch_size=-1):
        # create content in markdown
        text = '# Local download of the Unsplash Dataset\n\n'

        text += '![](https://drive.google.com/uc?export=view&id=1DZz--VgBa2dxYne4ATCbTUdNGY4W79S4)\n\n'

        text += '## 1 - Unsplash Dataset Access\n\n'

        text += '*Note: Request access at [Unsplash Dataset](https://unsplash.com/data)*\n\n'

        text += 'Dataset type: ' + self.dataset_type + '<br> \n'
        text += 'Name: ' + self.author_name + '\n\n'
                
        text+= 'Access date: ' + str(datetime.date.today()) + '\n\n'

        text += '## 2 - Dataset Parameters\n\n'

        text += 'Downloaded images: ' + str(self.get_number_images()) + '<br>' + '\n'
        text += 'Images in `cache.csv`: ' + str(len(urls)) + '<br>' + '\n'

        if batch_size > 0:
            num_batches = int(len(urls) / batch_size)
            text += '> Batches: ' + str(num_batches) + '<br>' + '\n'
            text += '> Batch size: ' + str(batch_size) + ' images per batch' + '<br>' + '\n\n'

        text += 'Dataset size: **' + str(np.round(self.get_used_space() / 1073741824, 2)) + '** GB' + '\n\n'
                
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