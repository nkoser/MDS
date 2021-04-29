import os
import glob

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.python.keras.models import Sequential

from autoe import Autoencoder


def get_data(base_dir):
    """
    Load the data (index 5 are the labels)
    :param base_dir: the directory where the data is stored
    :return: training and test data in a numpy list
    """
    train_path = os.path.join(base_dir, 'training')
    test_path = os.path.join(base_dir, 'testing')
    train_data = [np.load(os.path.join(train_path, f)) for f in os.listdir(train_path)]
    train_labels = train_data[5]
    test_data = [np.load(os.path.join(test_path, f)) for f in os.listdir(test_path)]
    test_labels = test_data[5]
    del train_data[5]
    del test_data[5]
    return train_data, train_labels, test_data, test_labels


def normalize_data(data):
    """

    :param data: numpy array with the data
    :return: normalize data to 0..1
    """
    for i in range(data.shape[2]):
        # normalize the data over the channels
        data[:, :, i] = MinMaxScaler().fit_transform(data[:, :, i])
    return data


if __name__ == '__main__':

    # ******************************************************************************************************************
    # ******************************* preprocessing ********************************************************************
    # ******************************************************************************************************************

    root = r'E:\Dokumente\MDS\bbh\bbh'
    train, train_Y, test, test_Y = get_data(root)
    norm_train_data = [normalize_data(d) for d in train]
    norm_test_data = [normalize_data(d) for d in test]

    # ******************************************************************************************************************
    # ******************************* feature extraction ***************************************************************
    # ******************************************************************************************************************

    latent_dim = 64
    features = np.zeros(shape=(len(norm_train_data), train_Y.shape[0], latent_dim))
    for k, data in enumerate(norm_train_data):
        autoencoder = Autoencoder(latent_dim, data.shape[1])
        autoencoder.compile(metrics=['accuracy'], optimizer='adam', loss=losses.MeanSquaredError())
        history = autoencoder.fit(data, data,
                                  epochs=30,
                                  shuffle=True,
                                  # validation_data=(x_test, x_test),
                                  verbose=0)
        plt.plot(history.history["loss"], label="Training Loss")
        plt.legend()
        plt.savefig("train_{}.png".format(k + 1))

        print(autoencoder.encoder(data).shape)
        features[k] = autoencoder.encoder(data)

    print(features.shape)
    print(features)
    print(np.unique(train_Y))
    num_classes = np.unique(train_Y).shape[0]
    features = features.reshape((-1, 9 * 64))

    model = Sequential()
    model.add(Dense(350, input_shape=features.shape, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    ## features = features.reshape((-1, 9, 64))
    ## vgg16 = VGG16(include_top=False,
    ##               weights=None,
    ##               input_shape=(9, 64, 1),
    ##               pooling='avg', )
    ##
    ## x = vgg16.output
    ## x = Flatten()(x)
    ## x = Dense(128, name='FC-Layer')(x)
    ## x = Dropout(0.2)(x)
    ## x = Activation('relu')(x)
    ## output = Dense(1, activation='sigmoid')(x)
    ##
    ## model = Model(vgg16.input, output)
    model.summary()

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=features,
                        y=train_Y,
                        batch_size=32,
                        epochs=40,
                        verbose=1,
                        validation_split=0.2)

    ## class Feature_NN(Model):
    ##     def __init__(self, input_size):
    ##         self.input_shape = input_size
    ##         super(Feature_NN, self).__init__()
    ##
    ##         self.encoder = tf.keras.Sequential([
    ##             layers.Dense(32, activation="relu"),
    ##             layers.Dense(16, activation="relu"),
    ##             layers.Dense(8, activation="relu")])
    ##
    ##         self.decoder = tf.keras.Sequential([
    ##             layers.Dense(16, activation="relu"),
    ##             layers.Dense(32, activation="relu"),
    ##             layers.Dense(80, activation="sigmoid")])
    ##
    ##     def call(self, x):
    ##         temp = []
    ##         for channel in x.shape(2):
    ##             encoded = self.encoder(x[:, :, channel])
    ##             temp.append(temp)
    ##             decoded = self.decoder(encoded)
    ##         return decoded
    ##
    ##
    ## auto_encoder = Feature_NN()
    ## auto_encoder.compile(optimizer='adam', loss='mae')
    ## # auto_encoder(norm_train_data[3][:, :, -1])
    ##
    ## history = auto_encoder.fit(norm_train_data[3][:, :, -1], norm_train_data[3][:, :, -1],
    ##                            epochs=100,
    ##                            batch_size=32,
    ##                            # validation_data=(test_data, test_data),
    ##                            shuffle=True)
    ##
    ## plt.plot(history.history["loss"], label="Training Loss")
    ## # plt.plot(history.history["val_loss"], label="Validation Loss")
    ## plt.legend()
    ## plt.savefig("test.png")
