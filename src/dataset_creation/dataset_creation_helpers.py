### dataset_creation_helpers.py ###
from matplotlib.pyplot import bar
import pandas as pd
import numpy as np
import os
from urllib import request
import cv2
from tqdm import tqdm
import datetime

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
    dirs = path.split(('\\'))
    joined_path = dirs[0]

    for dir in dirs[1:]:
        joined_path = os.path.join(joined_path, dir)
        if os.path.exists(joined_path) is False:
            os.mkdir(joined_path)

def retrieve_image_from_url(url):
    response = request.urlopen(url)

    code = response.getcode()
    img = np.array(bytearray(response.read()), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)

    return (code, img) 

class DatasetCreator:
    def __init__(self, path_in, path_out, sizes=(256, 64), dataset_size=2e4, download_ratio=1):
        self.path_in = path_in
        self.path_out = path_out

        self.sizes = np.sort(sizes)[::-1]

        self.dataset_size = dataset_size
        self.download_ratio = download_ratio

        self.images = np.array([])

        self.dataframe = pd.DataFrame()

        self.dataset_type = 'Lite'
        self.author_name = 'unknown'

    # loads 'photos.tsv000' from self.path_in into a dataframe
    def load_dataframe(self):
        self.dataframe = pd.read_csv(self.path_in+'\\photos.tsv000', sep='\t', header=0)

    # loads all the image urls into self.images
    # (for the future: here could be somekind of search algorithm for differing different types implemented)
    def search_images(self):
        self.images = np.array(self.dataframe.loc[:, 'photo_image_url'])

        if len(self.images) < self.dataset_size:
            print(bcolors.WARNING + 'Warning: The Unsplash Dataset contains just ' + str(len(self.images)) + ' images!' + bcolors.ENDC)
        else:
            self.images = self.images[0:int(self.dataset_size)]

    # downloads all the images from self.images and saves them in self.path_out
    def download_images(self):
        data_path = os.path.join(self.path_out, 'data')

        dir = 0
        sub_dir = 0

        for s in range(len(self.sizes)):
            make_dir(os.path.join(data_path, 'LOD_'+str(s)+'\\'+str(dir)+'\\'+str(sub_dir)))

        for i in tqdm(range(int(self.dataset_size*self.download_ratio))):
            if i % 1e4 == 0 and i!= 0:
                dir += 1
                sub_dir = 0
            elif i % 1e3 == 0 and i!= 0:
                sub_dir += 1
                for s in range(len(self.sizes)):
                    make_dir(os.path.join(data_path, 'LOD_'+str(s)+'\\'+str(dir)+'\\'+str(sub_dir)))

            size_suffix = '?ixid=123123123&fit=crop&w='+str(self.sizes[0])+'&h='+str(self.sizes[0])
            code, img = retrieve_image_from_url(self.images[i]+size_suffix)

            if code != 200:
                print(bcolors.WARNING + 'Warning: The Image ' + self.images[i] + ' could not be downloaded!' + bcolors.ENDC)
            else:
                img_index = int(i - dir*1e4 - sub_dir*1e3)

                for s in range(len(self.sizes)):
                    img_path = 'LOD_'+str(s)+'\\'+str(dir)+'\\'+str(sub_dir)+'\\image_'+str(img_index)+'.jpg'
                    cv2.imwrite(os.path.join(data_path, img_path), cv2.resize(img, (self.sizes[s], self.sizes[s])))

    # saves self.images in a .csv file
    def save_cache(self):
        df = pd.DataFrame(self.images)
        df.to_csv(self.path_out)

    def set_infos(self, dataset_type='Lite', author_name='unknown'):
        self.dataset_type = dataset_type
        self.author_name = author_name

    # creates a README.md file with information about the locally downloaded dataset
    def make_README(self):
        scriptDir = os.path.dirname(__file__)
        img_path = os.path.join(scriptDir, '..\\..\\cosmetic_images\\Unsplash_Mountains.png')
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        cv2.imwrite(self.path_out+'\\Unsplash_Mountains.png', img)

        text = '# Local download of the Unsplash Dataset\n\n'

        text += '![](./Unsplash_Mountains.png)\n\n'

        text += '## 1 - Unsplash Dataset Access\n\n'

        text += '*Note: Request access at [Unsplash Dataset](https://unsplash.com/data)*\n\n'

        text += 'Dataset type: ' + self.dataset_type + '<br> \n'
        text += 'Name: ' + self.author_name + '\n\n'
                
        text+= 'Access date: ' + str(datetime.date.today()) + '\n\n'

        text += '## 2 - Dataset Parameters\n\n'

        text += 'Downloaded images: ' + str(int(self.dataset_size*self.download_ratio)) + '<br> \n'
        text += 'Images in `cache.csv`:' + str(len(self.images)) + ' (' + str(len(self.images) - int(self.dataset_size*self.download_ratio)) + 'remaining)\n\n'
                
        text += '## 2 - Image Resolutions\n\n'

        text += '| Level of Detail             | Image Resolution |\n'
        text += '|-----------------------------|------------------|'

        for s in range(len(self.sizes)):
            text += '\n| LOD_'+str(s) + '                       | ('
            text += str(self.sizes[s]) + ' x ' + str(self.sizes[s]) + ')    |'

        file = open(self.path_out+'\\README.md', 'a')
        file.write(text)
        file.close()


dataset_creator = DatasetCreator('D:\\UNSPLASH Dataset Lite', 'D:\\UNSPLASH Dataset', dataset_size=25e3)

dataset_creator.load_dataframe()

dataset_creator.search_images()

dataset_creator.download_images()

dataset_creator.save_cache()

dataset_creator.set_infos(dataset_type='Lite', author_name='Lukas Hueglin')

dataset_creator.make_README()