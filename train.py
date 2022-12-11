# import necessary libraries
import os
import pathlib
import argparse
from imutils import paths
from preprocessing import *
from model import *
from evaluation import *
import tensorflow as tf
from tensorflow.keras import callbacks
tf.get_logger().setLevel('ERROR')


# argparse
parser = argparse.ArgumentParser(description='Process command line arguments.')
parser.add_argument("d", help="data folder path", type=pathlib.Path)
parser.add_argument("n", help="number of epochs", type=int)
parser.add_argument("-w", help="width of images", type=int, default=128)
parser.add_argument("-e", help="height of images", type=int, default=128)
parser.add_argument("-g", help="gray images", action='store_true')
args = parser.parse_args()



if __name__ == "__main__":
    # get image paths and split into train and test
    train_path, test_path = path_split(args.d)

    # Preprocessing and Labeling
    train_X, train_y = data_extractor(train_path, args.e, args.w, args.g)
    test_X, test_y = data_extractor(test_path, args.e, args.w, args.g)

    # if args.g is set to true, the images will be grayed out and the model will receive images with 1 channel
    if args.g == True:
        model = initialize_model(args.e, args.w, 1)
    elif args.g == False:
        model = initialize_model(args.e, args.w, 3)

    # Callbacks
    os.mkdir('MODEL_PATH')
    Callbacks = [
        callbacks.ModelCheckpoint('MODEL_PATH/', monitor='val_accuracy', save_best_only=True, verbose=0),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=4, verbose=0)]

    # compile the model
    model = compile_model(model) 
    # train the model   
    History = model.fit(train_X, train_y, epochs=args.n, batch_size=32, validation_data=(test_X, test_y), callbacks=Callbacks)
    
    # Model evaluation
    
    # plot train/validation loss/accuracy figures
    plot_history(History, epoch=args.n, metric='accuracy')
    plot_history(History, epoch=args.n, metric='loss')
    # print metrics
    print_evaluation_metrics(model, test_X, test_y)
    print_metrics(model, test_X, test_Y)
