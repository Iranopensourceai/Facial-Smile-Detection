import numpy as np
import os
import argparse
import cv2
from imutils import paths
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Layer,Conv2D,Activation,MaxPool2D,Dense,Flatten,Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow import keras
from imutils import paths
import cv2
import os
from sklearn.model_selection import train_test_split



def train_val_test_split(dataset_path, seed=432):
    """
    Splits images paths in the dataset to train and test 
    dataset_path: the path of dataset
    seed: the seed required to random shuffle files
    """
    imgpaths = list(paths.list_images(dataset_path))
    train_path, test_path = train_test_split(imgpaths, test_size=0.15, random_state=seed, shuffle=True)
    
    return train_path, test_path


def dataextractor(image_paths, img_height, img_width, gray=True):
    data=[]
    labels = []
#     imagepaths = list(paths.list_images(data_path))
    for imagepath in image_paths:
        image = cv2.imread(imagepath)
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(img_height, img_width),interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        label = imagepath.split(os.sep)[-2]
        label = int(label)
        labels.append(label)
        data.append(image)
    return np.array(data, dtype='float') / 255.0, np.array(labels)
    # splitting the data into train and test

# (train_X,test_X,train_y,test_y) = train_test_split(data,labels,test_size=0.2,random_state=123)


def augmentation(img, training=True):
    return keras.Sequential([
    preprocessing.RandomContrast(factor=0.3),
    preprocessing.RandomFlip(mode='horizontal'), # meaning, left-to-right
    preprocessing.RandomWidth(factor=0.15), # horizontal stretch
    preprocessing.RandomRotation(factor=0.20),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1)])(img, training)
    