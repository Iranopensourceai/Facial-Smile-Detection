# Importing necessary libraries 
from preprocessing import *
from tensorflow.keras import Sequential, layers
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers.experimental.preprocessing import Resizing


def initialize_model(imgs_height, imgs_width, n_channels):
    """
    imports the VGG model's top layers except input and classification layer, and set trainable to False
    then added the custom input and classifier layer
    :param imgs_height: input image height
    :param imgs_width: input image width
    :param n_channels: input image channels
    :return: a VGG model
    """
    vgg = VGG19(include_top=False, weights='imagenet', input_shape=(imgs_height, imgs_width, n_channels))

    for layer in vgg.layers:
        layer.trainable = False

    input_ = keras.Input(shape=(imgs_height, imgs_width, n_channels))
    x = augmentation_layer()(input_)
    x = Resizing(imgs_height, imgs_width)(x)

    vgg_layers = vgg.layers[1:]
    for layer in vgg_layers:
        x = layer(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    prediction = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(inputs=input_, outputs=prediction)


# defining a function for compile the model
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model
