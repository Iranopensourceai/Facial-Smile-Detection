# import necessary libraries
import os
import cv2
import pathlib
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
tf.get_logger().setLevel('ERROR')


# argparse
parser = argparse.ArgumentParser(description='Process command line arguments.')
parser.add_argument("p", help="test image path", type=pathlib.Path)
parser.add_argument("-w", help="width of image", type=int, default=128)
parser.add_argument("-e", help="height of image", type=int, default=128)
parser.add_argument("-g", help="gray images", action='store_true')
args = parser.parse_args()


# Pre-processing for test images
def test_preprocessing(img_path, gray=True):
    image = cv2.imread(img_path)
    if gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (args.e, args.w), interpolation=cv2.INTER_AREA)
    image = image / 255.0
    return np.expand_dims(image, axis=0)


if __name__ == "__main__":
    # Pre-processing
    test_image = test_preprocessing(args.p, args.g)

    # load model
    model = load_model('MODEL_PATH/model.h5')

    # prediction
    y_pred = model.predict(test_image)[0][0]

    if y_pred > 0.5:
        print("Smile:  {:.0%}".format(y_pred))
    else:
        print("non Smile:  {:.0%}".format(1.0 - y_pred))
