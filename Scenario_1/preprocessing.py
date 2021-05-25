import os
import glob

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.manifold import Isomap
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from plot_keras_history import plot_history
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import MaxPooling1D
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.python.layers.convolutional import Conv1D

from auto_conv1d_encoder import Autoencoder_Conv_80, Autoencoder_Conv_200, Autoencoder_Conv_268, Autoencoder_Conv_800
from autoe import Autoencoder
from plotting import plotting_history_1, customize_axis_plotting


def get_data(base_dir):
    """
    Load the data (index 5 are the labels)
    :param base_dir: the directory where the data is stored
    :return: training and test data in a numpy list
    """
    train_path = os.path.join(base_dir, 'training')
    test_path = os.path.join(base_dir, 'testing')
    train_data = [np.load(os.path.join(train_path, f)) for f in os.listdir(train_path)]
    label_idx = np.argmin(np.array([len(i.shape) for i in train_data]))
    train_labels = train_data[label_idx]
    del train_data[label_idx]
    test_data = [np.load(os.path.join(test_path, f)) for f in os.listdir(test_path)]
    label_idx = np.argmin(np.array([len(i.shape) for i in test_data]))
    test_labels = test_data[label_idx]
    del test_data[label_idx]
    return train_data, train_labels, test_data, test_labels


def normalize_data(d, dim=2):
    """
    :param dim:
    :param d: numpy array with the data
    :return: normalize data to 0..1
    """
    for i in range(d.shape[dim]):
        # normalize the data over the channels
        d[:, :, i] = MinMaxScaler().fit_transform(d[:, :, i])
    return d


def cropping(d):
    size = d.shape[1]
    crop = (size - 80) // 2
    model = Sequential()
    model.add(layers.Cropping1D(cropping=crop, ))
    x = model.predict(d)
    return x


if __name__ == '__main__':
    # ******************************************************************************************************************
    # ******************************* preprocessing ********************************************************************
    # ******************************************************************************************************************

    root = '/Users/niklaskoser/Documents/MDS/bbh'
    train, train_Y, test, test_Y = get_data(root)
    # cropped_train_data = [cropping(d) for d in train]
    norm_train_data = [normalize_data(d) for d in train]  # cropped_train_data]
    # cropped_test_data = [cropping(d) for d in test]
    norm_test_data = [normalize_data(d) for d in test]  # cropped_test_data]

    # ******************************************************************************************************************
    # ******************************* feature extraction ***************************************************************
    # ******************************************************************************************************************

    latent_dim = 5
    features = np.zeros(
        shape=(9, train_Y.shape[0], latent_dim))  # (len(norm_train_data), train_Y.shape[0], latent_dim))
    for k, data in enumerate(norm_train_data):
        input_size = data.shape[1]
        autoencoder = Sequential()
        if input_size == 80:
            autoencoder = Autoencoder_Conv_80(norm_train_data[k].shape, latent_dim)
        elif input_size == 200:
            autoencoder = Autoencoder_Conv_200(norm_train_data[k].shape, latent_dim)
        elif input_size == 268:
            autoencoder = Autoencoder_Conv_268(norm_train_data[k].shape, latent_dim)
        elif input_size == 800:
            autoencoder = Autoencoder_Conv_800(norm_train_data[k].shape, latent_dim)

        autoencoder.compile(metrics=[tf.keras.metrics.MeanSquaredError()], optimizer=tf.keras.optimizers.Adam(lr=0.001),
                            loss=losses.MeanSquaredError())
        history = autoencoder.fit(data, data,
                                  epochs=10,
                                  shuffle=True,
                                  batch_size=16,
                                  verbose=1,
                                  validation_split=0.2)

        plotting_history_1(history.history,
                           str(data.shape[1]) + "_autoencoder_{}_.png".format(k + 1),
                           f=customize_axis_plotting("loss"))
        print(autoencoder.encoder(data).shape)
        features[k] = autoencoder.encoder.predict(data)
        tf.keras.backend.clear_session()

    num_classes = np.unique(train_Y).shape[0]

    ## featuresr = []
    ## for k in norm_train_data:
    ##     embedding = Isomap(n_components=20)
    ##     X_transformed = embedding.fit_transform(k.reshape(k.shape[0], -1))
    ##     featuresr.append(X_transformed)

    ## features = np.stack(featuresr)
    features = features.reshape((-1, latent_dim, 9))
    print(features.shape)

    model = Sequential()
    model.Dense(256, activation="relu")
    model.Dense(128, activation="relu")
    model.Dense(num_classes, activation="softmax")
    ##model.add(layers.Conv1D(kernel_size=5, filters=16, activation='relu', padding="SAME"))
    ##model.add(layers.BatchNormalization())
    ##model.add(layers.MaxPool1D(2))
    ##model.add(layers.Conv1D(kernel_size=3, filters=64, activation='relu', padding='SAME'))
    ##model.add(layers.BatchNormalization())
    ### model.add(layers.MaxPool1D(2))
    ##model.add(layers.Conv1D(kernel_size=1, filters=128, activation='relu', padding="SAME"))
    ##model.add(layers.BatchNormalization())
    ##model.add(layers.Flatten())
    ##model.add(layers.Dense(128))
    ##model.add(Dense(num_classes, activation='softmax'))

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

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    features = normalize_data(features)
    print(np.max(features))
    history = model.fit(features,
                        y=to_categorical(train_Y),
                        batch_size=16,
                        epochs=200,
                        verbose=1,
                        validation_split=0.2)

    model.summary()
    plotting_history_1(history.history,
                       "final_classifier.png",
                       f=customize_axis_plotting("loss"))

    predictions = model.predict_classes(norm_test_data, verbose=2)
    print(predictions)
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
