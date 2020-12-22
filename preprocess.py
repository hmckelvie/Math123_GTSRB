import skimage
from skimage import io
import numpy as np
import os
import pandas as pd


IMG_SIZE = 48
FEATURES_FILE = 'preprocessed_features.np'

'''
traverse
param: path, the base path 
return: none
preprocesses each of the training images and saves them in the Preprocessed directory
'''
def traverse(path): 
    direct = os.listdir(path+"/Train")
    for d in direct:
        dir_path = path+"/Processed/"+ d
        images = os.listdir(path+"/Train/"+ d)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path) 
        for i in images :  
            img_path = dir_path +"/" + i
            src_path = path+"/Train/"+ d +"/" + i
            if not os.path.isfile(img_path) :
              fixed_img = preprocess_img(io.imread(src_path))
              io.imsave(img_path, img_as_ubyte(fixed_img))
'''
preprocess_img
param: the image, opened with skimage
return: the enhanced image
preprocesses the image to enhance the color and resolution
code is adapted from from https://software.intel.com/content/www/us/en/develop/articles/tutorial-building-and-training-the-traffic-image-model.html
'''
def preprocess_img(img):
    hsv = skimage.color.rgb2hsv(img)
    hsv[:,:,2] = skimage.exposure.equalize_hist(hsv[:,:,2])
    img = skimage.color.hsv2rgb(hsv)

    min_side = min(img.shape[:-1])
    center = img.shape[0]//2, img.shape[1]//2
    img = img[centre[0]-min_side//2:centre[0]+min_side//2,
              centre[1]-min_side//2:centre[1]+min_side//2,
              :]
    img = skimage.transform.resize(img, (IMG_SIZE, IMG_SIZE))
    img = np.rollaxis(img,-1, -1)
    return img
'''
load_preprocessed
param: path, the base path
returns: the labeled images where the labels are given from the input data, saves the features array to a file
creates the features array and saves to file and returns a dataframe of the expected labels for the output
'''
def load_preprocessed(path, file_name):
    features = []
    dirs = os.listdir(path)
    labeled_images = pd.DataFrame(columns=['path', 'expected_label'])
    for d in dirs:
        images = os.listdir(path+"/"+d)
        for img in images:
            curr_img = pd.DataFrame({'path': [path + "/" + d + "/"+ img], 'expected_label': [int(d)]})
            labeled_images = labeled_images.append(curr_img, ignore_index=True)
            img = io.imread(path + "/" + d + "/"+ img)
            features.append(img.flatten())
    arr_features = np.array(features)
    with open(file_name, 'wb') as file:
        np.save(file, arr_features)
    return labeled_images

