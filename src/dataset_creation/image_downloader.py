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

# retrieves one image from an url. The function is made to be called through a multiprocessing job,
# therefore it can't output anything to the command line. It returns a status, the image, and a string with print messages.
def retrieve_image_from_url(url, max_size):
    print_str = ''

    try:
        # download image
        size_suffix = '?ixid=123123123&fit=crop&w='+str(max_size)+'&h='+str(max_size)
        response = requests.get(url+size_suffix, stream=True)

        # decode response to opencv image
        img = np.array(bytearray(response.content), dtype="uint8")
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        # raise error if img is empty (there are some urls that just crash up everything without it)
        x = img[0]

        # OK indicates, that everything worked great
        return ('ok', img, print_str)

    # requests exeptions
    except requests.exceptions.HTTPError as e:
        print_str += (TColors.FAIL + "HTTPError: " + str(e) + '\n' + TColors.WARNING + "Note: URL: " + url + TColors.ENDC)
    except requests.exceptions.Timeout as e:
        print_str += (TColors.FAIL + "Timeout: " + str(e) + '\n' + TColors.WARNING + "Note: The image will be downloaded again later, URL: " + url + TColors.ENDC)
        return ('repeat', np.array([]))
    except requests.exceptions.TooManyRedirects as e:
        print_str += (TColors.FAIL + "TooManyRedirects: " + str(e) + '\n' + TColors.WARNING + "Note: URL: " + url + TColors.ENDC)
    except requests.exceptions.RequestException as e:
        print_str += (TColors.FAIL + "RequestException: " + str(e) + '\n' + TColors.WARNING + "Note: URL: " + url + TColors.ENDC)
        
    # catch rest I didn't think could happen :)
    except:
        print_str += (TColors.FAIL + "Something else failed!" + '\n' + TColors.WARNING + "Note: URL: " + url + TColors.ENDC)

    # False indicates, that the url has to be downloaded again
    return ('crash', np.array([]), print_str)

# download multiple images. This function is also made to be called as a multiprocessing job,
# that's why the images, and print messages are returned as a list instead of a return value.
def retrieve_images(urls, img_list, print_list, max_size, repeated=False):
    repeated_urls = []

    # iterates through all urls
    for url in urls:
        # retrieves one of them
        (status, img, print_str) = retrieve_image_from_url(url, max_size)
        if status == 'ok' and True:
            img_list.append(img)
        elif status == 'repeat' and repeated is False:
            repeated_urls.append(img)

        if print_str != '':
            print_list.append(print_str)

    # calls it self it there are images, which need to be downloaded again.
    if len(repeated_urls) != 0:
        retrieve_images(repeated_urls, img_list, print_list, max_size, repeated=True)

# (with help from: https://towardsdatascience.com/multithreading-multiprocessing-python-180d0975ab29
# and from: https://stackoverflow.com/questions/42490368/appending-to-the-same-list-from-different-processes-using-multiprocessing)

# This function retrieves a list of images via multiprocessing
def retrieve_images_multiproc(urls, max_size, num_proc):
    split_urls = np.split(urls, num_proc)

    with multiprocessing.Manager() as manager:
        jobs = []
        img_list = manager.list()
        print_list = manager.list()

        for i in range(num_proc):
            process = multiprocessing.Process(
                target=retrieve_images,
                args=(split_urls[i], img_list, print_list, max_size)
            )
            jobs.append(process)

        # start end end jobs
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()

        # output the print messages
        for str in print_list:
            print(str)

        return np.array(img_list)
