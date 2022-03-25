### dataset_creation_helpers.py ###
import pandas as pd
import numpy as np

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
        None

    # downloads all the images from self.images and saves them in self.path_out
    def download_images(self):
        None

    # saves self.images in a .csv file
    def save_cache(self):
        None

    # creates a README.md file with information about the locally downloaded dataset
    def make_README(self):
        None


dataset_creator = DatasetCreator('D:/UNSPLASH Dataset Lite', 'D:/UNSPLASH Dataset')

dataset_creator.load_dataframe()

dataset_creator.search_images()