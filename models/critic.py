from math import pi

import tensorflow as tf
import numpy as np
from models.proxy_net import ProxyNet
from models.submodels import MapFeaturesProcessor, FeatureExtractorLayer, ActionProcessor

from utils.constants import Car
from utils.crucial_points import calculate_car_crucial_points
from utils.distances import dist, path_dist, if_inside, path_line_dist, path_dist_cp
from utils.poly5 import curvature, params
from utils.utils import _calculate_length, Rot
from matplotlib import pyplot as plt

tf.enable_eager_execution()


class Critic(ProxyNet):

    def __init__(self):
        super(Critic, self).__init__()

        n = 128
        self.map_processing = MapFeaturesProcessor(n)
        self.preprocessing_stage = FeatureExtractorLayer(n)
        self.action_processor = ActionProcessor(n)

        act = tf.keras.activations.relu
        self.fc = [
            tf.keras.layers.Dense(n, activation=act),
            tf.keras.layers.Dense(64, activation=act),
            tf.keras.layers.Dense(1, activation=None),
        ]

    def call(self, state, action):
        p0, pk, free_space, path = state
        x0, y0, th0, ddy0 = tf.unstack(p0, axis=-1)
        xk, yk, thk = tf.unstack(pk, axis=-1)

        W = 20.
        H = 20.

        map_features = self.map_processing(free_space)

        inputs = tf.stack([x0 / W, y0 / H, th0 / (2 * pi), ddy0, xk / W, yk / H, thk / (2 * pi)], -1)

        state_features = self.preprocessing_stage(inputs)
        action_features = self.action_processor(action)
        x = tf.concat([state_features, map_features, action_features], -1)
        for l in self.fc:
            x = l(x)
        return x
