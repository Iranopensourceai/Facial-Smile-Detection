import cv2
import os
from imutils import paths
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import *
 


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
    """
    This function read and resize input images,
    set the grayscale of the images
    label them, then return two arrays of data and labels
    """
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
    

 
def augmentation(img, training=True):
    """
    This function is used for data augumentation
    """
    return keras.Sequential([
    RandomContrast(factor=0.5),
    RandomFlip(mode='horizontal'), # meaning, left-to-right
    RandomWidth(factor=0.15), # horizontal stretch
    RandomRotation(factor=0.20),
    RandomTranslation(height_factor=0.1, width_factor=0.1)])(img, training)



if __name__ == "main":
  train_path, val_path, test_path = train_val_test_split(dataset_path)

  train_X, train_y =dataextractor(train_path)
  val_X, val_y = dataextractor(val_path)
  test_X, test_y = dataextractor(test_path)

  ex = train_X[100]

  plt.figure(figsize=(10,10))
  for i in range(16):
      image = augmentation(ex)
      plt.subplot(4, 4, i+1)
      plt.imshow(tf.squeeze(image) )
      plt.axis('off')
  plt.show()