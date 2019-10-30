import tensorflow as tf


class MapAE(tf.keras.Model):
    def __init__(self):
        super(MapAE, self).__init__()
        self.encoder = [
            tf.keras.layers.Conv2D(16, 3, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(16, 3, strides=(2, 2), padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(16, 3, strides=(2, 2), padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2D(32, 3, strides=(2, 2), padding='same', activation=tf.keras.activations.relu),
        ]

        self.decoder = [
            tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2DTranspose(32, 3, 2, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2DTranspose(16, 3, 2, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2DTranspose(16, 3, 2, padding='same', activation=tf.keras.activations.relu),
            tf.keras.layers.Conv2DTranspose(2, 3, 2, padding='same', activation=None),
        ]

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.encoder:
            x = layer(x)
        for layer in self.decoder:
            x = layer(x)
        return x

    def encode(self, inputs, training=None):
        x = inputs
        for layer in self.encoder:
            x = layer(x)
        return x
