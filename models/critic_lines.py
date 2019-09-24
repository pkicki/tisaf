from math import pi
import tensorflow as tf
tf.enable_eager_execution()


class CriticNetwork(tf.keras.Model):
    def __init__(self):
        super(CriticNetwork, self).__init__()
        activation = tf.keras.activations.tanh
        kernel_init_std = 1.0
        self.features = [
            tf.keras.layers.Dense(256, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(256, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            #tf.keras.layers.Dense(256, activation,
            #                      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            #tf.keras.layers.Dense(256, activation,
            #                      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            #tf.keras.layers.Dense(256, activation,
            #                      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            #tf.keras.layers.Dense(256, activation,
            #                      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(1),
        ]

    def call(self, x, y, xk, yk, action, training=None):
        ex = (xk - x) / 15.
        ey = (yk - y) / 15.
        inputs = tf.concat([ex, ey, action], -1)
        x = inputs
        for layer in self.features:
            x = layer(x)
        return x

