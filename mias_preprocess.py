from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.image as matlabimg
import re
import pandas as pd
import os
from sklearn.feature_extraction import image

rng = np.random.RandomState(0)
patch_size = (64, 64)
global_counter = 0

# read csvFile
data_train = pd.read_csv('./mias/mias.csv')
abnormal = data_train[data_train.abnormality_class != 'NORM']
normal = data_train[data_train.abnormality_class == 'NORM']


# data_train
# HELPER FUNCTIONS
# data_train[data_train.abnormality_class == 'NORM']["abnormality_class"]

# Function for obtaining center crops from an image
def crop_center(x, crop_w, crop_h):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return x[j:j + crop_h, i:i + crop_w]


# Function for reading PGM files
def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))

#
# images = []
# for i, row in normal.iterrows():
#     images.append(read_pgm('./mias/pgm/' + row['reference_number'] + '.pgm'))
#
# j = 0;
# for i, row in normal.iterrows():
#     images[j].setflags(write=1)
#     if (int(row['reference_number'][-3:]) % 2 == 0):
#         images[j][:324, 700:1024] = np.zeros((324, 324))
#     else:
#         images[j][:324, :324] = np.zeros((324, 324))
#         matlabimg.imsave('./mias/png/' + row['reference_number'] + '.png', images[j], vmin=0, vmax=255, cmap='gray')
#     j += 1


def generate_patches(input_image):
    global global_counter
    input_image = crop_center(input_image, 384, 384)
    patches = image.extract_patches_2d(input_image, patch_size, max_patches=200,
                                       random_state=rng)
    for counter, i in enumerate(patches):
        if np.any(i):
            matlabimg.imsave('./data/mias200/' + str(global_counter) + '.jpg', i, cmap='gray')
            global_counter += 1


# images = []
# arr = os.listdir(os.getcwd() + "/mias/png/")
# for img in arr:
#     images.append(matlabimg.imread(os.getcwd() + "/mias/png/" + img))
#
# generate_patches(images[0])
for counter, image_full in enumerate(images):
    generate_patches(image_full)



# img = matlabimg.imread("./data/mias/87.jpg")
#
# print(img)
