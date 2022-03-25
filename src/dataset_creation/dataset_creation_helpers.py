### dataset_creation_helpers.py ###
from matplotlib.pyplot import bar
import pandas as pd
import numpy as np
import os
from urllib import request
import cv2
from tqdm import tqdm

# colors for terminal output
# (from: https://stackoverflow.com/questions/287871/how-to-print-colored-text-to-the-terminal)
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# creates a new directory if it does not exists
def make_dir(path):
    if os.path.exists(path) is False:
        os.mkdir(path)

def retrieve_image_from_url(url):
    response = request.urlopen(url)

    code = response.getcode()
    img = np.array(bytearray(response.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    return (code, img) 

class DatasetCreator:
    def __init__(self, path_in, path_out, sizes=(64, 256), dataset_size=2e4, download_ratio=1):
        self.path_in = path_in
        self.path_out = path_out

        self.sizes = sizes

        self.dataset_size = dataset_size
        self.download_ratio = download_ratio

        self.images = np.array([])

        self.dataframe = pd.DataFrame()

    # loads 'photos.tsv000' from self.path_in into a dataframe
    def load_dataframe(self):
        self.dataframe = pd.read_csv(self.path_in+'/photos.tsv000', sep='\t', header=0)

    # loads all the image urls into self.images
    # (for the future: here could be somekind of search algorithm for differing different types implemented)
    def search_images(self):
        self.images = np.array(self.dataframe.loc[:, 'photo_image_url'])

    # downloads all the images from self.images and saves them in self.path_out
    def download_images(self):
        make_dir(self.path_out)
        
        data_path = os.path.join(self.path_out, 'data')
        make_dir(data_path)

        if len(self.images) < self.dataset_size:
            print(bcolors.WARNING + 'Warning: The Unsplash Dataset contains just ' + str(len(self.images)) + ' images!' + bcolors.ENDC)

        dir = 0
        sub_dir = 0
        make_dir(os.path.join(data_path, '0'))
        make_dir(os.path.join(data_path, '0\\0'))

        for i in tqdm(range(int(self.dataset_size))):
            if i % 1e4 == 0 and i!= 0:
                dir += 1
                sub_dir = 0
                make_dir(os.path.join(data_path, str(dir)))
            elif i % 1e3 == 0 and i!= 0:
                sub_dir += 1
                make_dir(os.path.join(data_path, str(dir)+'\\'+str(sub_dir)))

            size_suffix = '?ixid=123123123&fit=crop&w='+str(np.amax(self.sizes))+'&h='+str(np.amax(self.sizes))
            code, img = retrieve_image_from_url(self.images[i]+size_suffix)

            if code != 200:
                print(bcolors.WARNING + 'Warning: The Image ' + self.images[i] + ' could not be downloaded!' + bcolors.ENDC)
            else:
                img_index = int(i - dir*1e4 - sub_dir*1e3)

                for s in range(len(self.sizes)):
                    img_path = str(dir)+'\\'+str(sub_dir)+'\\image_'+str(img_index)+'_LOD_'+str(s)+'.jpg'
                    cv2.imwrite(os.path.join(data_path, img_path), cv2.resize(img, (self.sizes[s], self.sizes[s])))

    # saves self.images in a .csv file
    def save_cache(self):
        None

    # creates a README.md file with information about the locally downloaded dataset
    def make_README(self):
        None


dataset_creator = DatasetCreator('D:/UNSPLASH Dataset Lite', 'D:/UNSPLASH Dataset', dataset_size=25e3)

dataset_creator.load_dataframe()

dataset_creator.search_images()

dataset_creator.download_images()