# import necessary libraries
import os
import cv2
import glob
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers.experimental import preprocessing


def get_images_path(dataset_path):
    """
    Extract image paths
    :param dataset_path: dataset path
    :return: list of image paths
    """
    list_path = []
    for img_path in glob.glob(f'/{dataset_path}/*/*'):
        list_path.append(img_path)
    return list_path


def path_split(dataset_path, seed=432):
    """
    Extracts image paths existing in the dataset and then splits to train, validation and test sets.
    :param dataset_path: the path of dataset
    :param seed: the seed required to random shuffle files
    :return: 
    """
    img_paths = get_images_path(dataset_path)
    train_path, test_path = train_test_split(img_paths, test_size=0.15, random_state=seed, shuffle=True)
    return train_path, test_path


def data_extractor(image_paths, img_height, img_width, gray=True):
    """
    This function reads the image existing in the input path and 
    doing some preprocessing operations on it.besides extracts the image label them.
    :param image_paths: the input image path
    :param img_height: resized image height
    :param img_width: resized image width
    :param gray: if True set image gray scale else not
    :return: image and label arrays
    """
    data = []
    labels = []
    for img_path in image_paths:
        image = cv2.imread(img_path)
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image,(img_height, img_width),interpolation=cv2.INTER_AREA)
        label = img_path.split(os.sep)[-2]
        label = int(label)
        labels.append(label)
        data.append(image)
    if gray:
        return np.expand_dims(data, axis=-1)/255.0, np.array(labels)
    else:
        return np.array(data)/255.0, np.array(labels)


def augmentation_layer():
    """
    Creates Augmentation layer to produce extra data for boosting classifier.
    Including in the model to use GPU instead of CPU for speeding up computations.
    :return: Fake data
    """
    return Sequential([
        preprocessing.RandomFlip('horizontal'),
        preprocessing.RandomContrast(factor=0.3),
        preprocessing.RandomWidth(factor=0.15),
        preprocessing.RandomRotation(factor=0.20),
        preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1)
    ])
