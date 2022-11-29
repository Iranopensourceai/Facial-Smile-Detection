import tensorflow as tf 
import os
import cv2
from tensorflow import keras
import numpy as np
from skimage import io, transform
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing import ImageDataGenerator
from imutils import paths

train_folder = '.../datasets/train_folder/'
test_folder = '.../datasets/test_folder/'

def dataExtractor(path, height=32, width=32):
    data = []
    labels = []
    image_paths = list(paths.list_images(path))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(height,width), interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        label = image_path.split(os.sep)[-2]
        label = int(label)
        labels.append(label)
        data.append(image)
    return np.array(data, dtype='float')/255.0, np.array(labels)

train_x, train_y = dataExtractor(train_folder)
test_x, test_y = dataExtractor(test_folder)


height = 32
width = 32
depth =1
classes=2

input_shape = (width,height,depth)
 
