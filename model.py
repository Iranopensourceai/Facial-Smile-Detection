# Importing necessary libraries 
from tensorflow.keras import Sequential, layers


# defining the Convolutional Neural Network
def initialize_model(imgs_height, imgs_width, n_channels):
    model = Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(imgs_height, imgs_width, n_channels), padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding='same'))
    model.add(layers.MaxPool2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", padding='same'))
    model.add(layers.MaxPool2D(pool_size=(3, 3)))
    model.add(layers.Flatten())
    model.add(layers.Dense(50, activation='relu'))
    model.add(layers.Dropout(rate=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# defining a function for compile the model
def compile_model(model):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model