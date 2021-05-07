import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import math


class Autoencoder_Conv1D(Model):

    def __init__(self, input_shape, latent_dim):
        self.num_neurons = math.ceil(((input_shape[1] / 2) / 2) / 2)
        super(Autoencoder_Conv1D, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Conv1D(kernel_size=3, filters=8, activation='relu', padding='SAME'),
            layers.MaxPool1D(2, padding='SAME'),
            layers.Flatten(),
            layers.Dense(units=latent_dim)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(units=self.num_neurons),
            layers.Reshape(target_shape=(-1, 1)),
            layers.Conv1D(kernel_size=3, filters=8, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=2, filters=16, activation='relu', ) if input_shape[1] == 268
            else layers.Conv1D(kernel_size=3, filters=16, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=32, activation='relu', padding='SAME'),
            layers.UpSampling1D(size=2),
            layers.Conv1D(kernel_size=3, filters=3, activation='relu', padding='SAME'),
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
