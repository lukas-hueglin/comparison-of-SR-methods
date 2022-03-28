### image_downloading.py ###
import multiprocessing
import requests

import cv2
import numpy as np

from enum import Enum

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

class Status(Enum):
    OK = 1
    CRASH = 2
    REPEAT = 3

def download_image_from_url(url, max_size):
    try:
        # download image
        size_suffix = '?ixid=123123123&fit=crop&w='+str(max_size)+'&h='+str(max_size)
        response = requests.get(url+size_suffix, stream=True)

        # decode response to opencv image
        img = np.array(bytearray(response.content), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # OK indicates, that everything worked great
        return (Status.OK, img)

    # requests exeptions
    except requests.exceptions.HTTPError as e:
        print(TColors.FAIL + "HTTPError: " + str(e) + '\n' + TColors.WARNING
                + "Note: URL: " + url + TColors.ENDC)
    except requests.exceptions.Timeout as e:
        print(TColors.FAIL + "Timeout: " + str(e) + '\n' + TColors.WARNING
                + "Note: The image will be downloaded again later, URL: " + url + TColors.ENDC)
        return (Status.REPEAT, np.array([]))
    except requests.exceptions.TooManyRedirects as e:
        print(TColors.FAIL + "TooManyRedirects: " + str(e) + '\n' + TColors.WARNING
                + "Note: URL: " + url + TColors.ENDC)
    except requests.exceptions.RequestException as e:
        print(TColors.FAIL + "RequestException: " + str(e) + '\n' + TColors.WARNING
                + "Note: URL: " + url + TColors.ENDC)
        
    # catch rest I didn't think could happen :)
    except:
        print(TColors.FAIL + "Something else failed!" + TColors.ENDC)

    # False indicates, that the url has to be downloaded again
    return (Status.CRASH, np.array([]))

# download multiple images
def download_images(urls, img_list, max_size, repeated=False):
    repeated_urls = []

    for url in urls:
        (status, img) = download_image_from_url(url, max_size)
        if status == Status.OK:
            img_list.append(img)
        elif status == Status.REPEAT and repeated is False:
            repeated_urls.append(img)

    if len(repeated_urls) != 0:
        download_images(repeated_urls, img_list, repeated=True)

# (with help from: https://towardsdatascience.com/multithreading-multiprocessing-python-180d0975ab29
# and from: https://stackoverflow.com/questions/42490368/appending-to-the-same-list-from-different-processes-using-multiprocessing)
def download_images_multiproc(urls, max_size, num_proc):
    split_urls = np.split(urls, num_proc)

    with multiprocessing.Manager() as manager:
        jobs = []
        img_list = manager.list()

        for i in range(num_proc):
            process = multiprocessing.Process(
                target=download_images,
                args=(split_urls[i], img_list, max_size)
            )
            jobs.append(process)

        # start end end jobs
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()

        return np.array(img_list)
