import cv2 as cv2
import numpy as np
import glob


# function to retrieve and reshape training and test sets

def retrieve_from(path, label):
    y = []
    x = []

    for idx, i in enumerate(label):
        for im in glob.glob(path + '/' + i + '/*.jpg', recursive=True):
            # color conversion to black and white images
            x.append(cv2.cvtColor(cv2.imread(im), cv2.COLOR_BGR2GRAY))
            y.append(idx)

    return x, y

#function to reshape the data set in an appropriate way

def reshape(x, width, height, channel):
    # Using anisotropic rescaling, resize the images to 64x64 in order to feed them to the network
    x = np.array([cv2.resize(im, (width, height), interpolation=cv2.INTER_AREA) for im in x])
    x = np.array([np.reshape(im, (width, height, channel)) for im in x])

    # normalize the images
    x = x / 255

    return x
