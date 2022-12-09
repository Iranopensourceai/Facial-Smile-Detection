# import necessary libraries
import os
import cv2
import argparse
import numpy as np
from imutils import paths
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers.experimental import preprocessing


# train test split
def path_split(dataset_path, seed=432):
    """
    Splits images paths in the dataset to train and test 
    dataset_path: the path of dataset
    seed: the seed required to random shuffle files
    """
    imgpaths = list(paths.list_images(dataset_path))
    train_path, test_path = train_test_split(imgpaths, test_size=0.15, random_state=seed, shuffle=True)
    
    return train_path, test_path


# Preprocessing and Labeling
def dataextractor(image_paths, img_height, img_width, gray=True):
    data=[]
    labels = []
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


# Data augmentation layer
def augmentation_layer(imgs_height, imgs_width, n_channels):
    return tf.keras.Sequential([                                    
    tf.keras.layers.RandomFlip('horizontal', input_shape=(imgs_height, imgs_width, n_channels)),
    preprocessing.RandomContrast(factor=0.3),
    preprocessing.RandomWidth(factor=0.15),
    preprocessing.RandomRotation(factor=0.20),
    preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1)])
