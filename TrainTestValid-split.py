#!/usr/bin/env python
# coding: utf-8
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



dataset_path = "E:\\smile-detection\\DataPreprocessing-Smile\\Data\\datasets"



def train_val_test_split(dataset_path, seed=432):
    """
        Splits images paths in the dataset to train, validation and test 
    dataset_path: the path of dataset
    seed: the seed required to random shuffle files
    """
    imgpaths = list(paths.list_images(dataset_path))
    train_val_path, test_path = train_test_split(imgpaths, test_size=0.1, random_state=seed, shuffle=True)
    train_path, validation_path = train_test_split(train_val_path, test_size=0.2)
    
    return train_path, validation_path, test_path


def dataextractor(image_paths,height=32,width=32):
    data=[]
    labels = []
    for imagepath in image_paths:
        image = cv2.imread(imagepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(height, width),interpolation=cv2.INTER_AREA)
        image = img_to_array(image)
        label = imagepath.split(os.sep)[-2]
        label = int(label)
        labels.append(label)
        data.append(image)
    return np.array(data, dtype='float') / 255.0, np.array(labels)


