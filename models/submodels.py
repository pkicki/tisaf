from math import pi

import tensorflow as tf
import numpy as np

from utils.constants import Car
from utils.crucial_points import calculate_car_crucial_points
from utils.distances import dist, path_dist, if_inside, path_line_dist, path_dist_cp
from utils.poly5 import curvature, params
from utils.utils import _calculate_length, Rot
from matplotlib import pyplot as plt

tf.enable_eager_execution()


class EstimatorLayer(tf.keras.Model):
    """
    Parameter estimator layer
    """

    def __init__(self, activation=tf.keras.activations.tanh, kernel_init_std=0.1, bias=0.0, mul=1., pre_mul=1.,
                 pre_bias=0.0):
        super(EstimatorLayer, self).__init__()
        self.bias = tf.Variable(bias, trainable=True, name="bias")
        self.mul = mul
        self.pre_mul = pre_mul
        self.pre_bias = pre_bias
        self.activation = activation
        self.features = [
            tf.keras.layers.Dense(128, tf.nn.tanh,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(64, tf.nn.tanh,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(64, tf.nn.tanh,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
        ]
        self.out = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std))

    def call(self, inputs):
        x = inputs
        for layer in self.features:
            x = layer(x)
        x = self.out(x)
        x *= self.pre_mul
        x += self.pre_bias
        x = self.activation(x)
        x *= self.mul
        x += self.bias
        return x


class FeatureExtractorLayer(tf.keras.Model):
    """
    Feature exrtactor layer
    """

    def __init__(self, num_features, activation=tf.keras.activations.tanh, kernel_init_std=0.1):
        super(FeatureExtractorLayer, self).__init__()
        self.features = [
            tf.keras.layers.Dense(64, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(num_features, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(num_features, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(num_features, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            # tf.keras.layers.Dense(num_features, activation),
        ]
        # self.fc = tf.keras.layers.Dense(num_features, activation)

    def call(self, inputs):
        x = inputs
        for layer in self.features:
            x = layer(x)
        # x = self.fc(x)
        return x


class ActionProcessor(tf.keras.Model):
    """
    """

    def __init__(self, num_features, activation=tf.keras.activations.tanh):
        super(ActionProcessor, self).__init__()
        self.features = [
            tf.keras.layers.Dense(64, activation),
            tf.keras.layers.Dense(num_features, activation),
            tf.keras.layers.Dense(num_features, activation),
        ]

    def call(self, inputs):
        x = inputs
        for layer in self.features:
            x = layer(x)
        return x

class MapFeaturesProcessor(tf.keras.Model):
    def __init__(self, num_features):
        super(MapFeaturesProcessor, self).__init__()
        self.num_features = 32
        self.point_processor = [
            tf.keras.layers.Dense(32, tf.keras.activations.tanh),
            tf.keras.layers.Dense(4 * self.num_features, tf.keras.activations.tanh),
        ]

        self.features = [
            # tf.keras.layers.Dense(32, tf.keras.activations.tanh),
            tf.keras.layers.Dense(64, tf.keras.activations.tanh),
            # tf.keras.layers.Dense(64, tf.keras.activations.tanh),
            tf.keras.layers.Dense(num_features, tf.keras.activations.tanh),
        ]

    def call(self, inputs, training=None):
        x = inputs
        bs = x.shape[0]
        n_quad = x.shape[1]
        n_points = x.shape[2]
        for layer in self.point_processor:
            x = layer(x)
        # x = tf.reshape(x, (bs, n_quad, n_points, self.num_features, 2, 2))
        x = tf.reshape(x, (bs, n_quad, n_points, self.num_features, 4))
        a, b, c, d = tf.unstack(x, axis=2)
        x = a[:, :, :, 0] * b[:, :, :, 1] * c[:, :, :, 2] * d[:, :, :, 3] \
            + b[:, :, :, 0] * c[:, :, :, 1] * d[:, :, :, 1] * a[:, :, :, 3] \
            + c[:, :, :, 0] * d[:, :, :, 1] * a[:, :, :, 1] * b[:, :, :, 3] \
            + d[:, :, :, 0] * a[:, :, :, 1] * b[:, :, :, 1] * c[:, :, :, 3]
        # mul = a @ b @ c @ d
        # x = tf.trace(mul)
        for layer in self.features:
            x = layer(x)
        x = tf.reduce_sum(x, 1)
        return x
