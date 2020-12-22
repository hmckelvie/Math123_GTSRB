import skimage
from skimage import io
from skimage import segmentation
import os
IMG_SIZE = 48

'''
preprocess_segmentation
param: path, the current path
       num_cluster, the number of clusters used in the image segmentation
returns: none
side effects: adds preprocessed segmented images to directory
for each preprocessed images, performs image segmentation with K-Means and the
  number of clusters passed in
'''
def preprocess_segmentation(path, num_cluster):
    out_dir = path+("/Processed_Segmented_%d"%(num_cluster)) 
    direct = os.listdir(path+"/Processed")
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir) 
    for d in direct:
        dir_path = path+ ("/Processed_Segmented_%d/"%(num_cluster)) + d
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path) 
        images = os.listdir(path+"/Processed/" + d)
        for i in images :  
            img_path = dir_path +"/" + i
            src_path = path+"/Processed/"+ d +"/" + i
            if not os.path.isfile(img_path) :
              fixed_img = img_seg(src_path, num_cluster)
              fixed_img = skimage.img_as_ubyte(fixed_img)
              io.imsave(img_path, fixed_img)

'''
img_seg
param: src_path, the path to the image
       num_cluster, the number of clusters used in the image segmentation
returns: the segmented image
performs image segmentation using K-Means clustering algorithm, 
resizes image to standard size 
'''
def img_seg(src_path, num_cluster):
    img = io.imread(src_path)
    segmented = segmentation.slic(img, n_segments=num_cluster)
    seg = skimage.color.label2rgb(segmented, img, kind='avg')   
    seg = skimage.transform.resize(seg, (IMG_SIZE, IMG_SIZE))
    return seg
