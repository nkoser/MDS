import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import math


class Autoencoder_Conv_80(Model):

    def __init__(self, input_shape, latent_dim):
        self.num_neurons = math.ceil(((input_shape[1] / 2) / 2) / 2)
        super(Autoencoder_Conv_80, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=1, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Flatten(),
            layers.Dense(units=latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(10),
            layers.Reshape(target_shape=(-1, 1)),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding="SAME"),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=3, activation='relu', padding='SAME'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        self.encoder.summary()
        decoded = self.decoder(encoded)
        self.decoder.summary()
        return decoded


class Autoencoder_Conv_200(Model):

    def __init__(self, input_shape, latent_dim):
        self.num_neurons = math.ceil(((input_shape[1] / 2) / 2) / 2)
        super(Autoencoder_Conv_200, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=1, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Flatten(),
            layers.Dense(units=latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(25),
            layers.Reshape(target_shape=(-1, 1)),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=3, activation='relu', padding='SAME'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class Autoencoder_Conv_268(Model):

    def __init__(self, input_shape, latent_dim):
        self.num_neurons = math.ceil(((input_shape[1] / 2) / 2) / 2)
        super(Autoencoder_Conv_268, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=1, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Flatten(),
            layers.Dense(units=latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(34),
            layers.Reshape(target_shape=(-1, 1)),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=2, filters=16, activation='relu', ),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=3, activation='relu', padding='SAME'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Autoencoder_Conv_800(Model):

    def __init__(self, input_shape, latent_dim):
        self.num_neurons = math.ceil(((input_shape[1] / 2) / 2) / 2)
        super(Autoencoder_Conv_800, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=64, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=1, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Flatten(),
            layers.Dense(units=latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(25),
            layers.Reshape(target_shape=(-1, 1)),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=64, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=3, activation='relu', padding='SAME'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded