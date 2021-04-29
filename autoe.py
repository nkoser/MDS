import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


class Autoencoder(Model):

    def __init__(self, latent_dim, long_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(latent_dim, activation='relu'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(long_dim*3, activation='sigmoid'),
            layers.Reshape((long_dim, 3))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
