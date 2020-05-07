from math import pi
import tensorflow as tf
from models.proxy_net import ProxyNet
from models.submodels import MapFeaturesProcessor, FeatureExtractorLayer, EstimatorLayer

tf.enable_eager_execution()


class PlanningNetwork(ProxyNet):
    def __init__(self):
        super(PlanningNetwork, self).__init__()

        n = 64

        self.map_processing = MapFeaturesProcessor(64)

        self.preprocessing_stage = FeatureExtractorLayer(n)
        self.x_est = EstimatorLayer(tf.nn.sigmoid, mul=10., bias=0.1, kernel_init_std=0.1, pre_bias=-1., pre_mul=1.0)
        self.y_est = EstimatorLayer(tf.identity)
        self.dy_est = EstimatorLayer(tf.identity)
        self.ddy_est = EstimatorLayer(tf.identity)
        self.last_ddy_est = EstimatorLayer(tf.identity)

    def call(self, data):
        p0, pk, free_space, path = data
        x0, y0, th0, ddy0 = tf.unstack(p0, axis=-1)
        xk, yk, thk = tf.unstack(pk, axis=-1)

        W = 20.
        H = 20.

        #map_features = self.map_processing(free_space)

        inputs = tf.stack([x0 / W, y0 / H, th0 / (2 * pi), ddy0, xk / W, yk / H, thk / (2 * pi)], -1)

        features = self.preprocessing_stage(inputs)
        #features = tf.concat([features, map_features], -1)

        x = self.x_est(features)
        y = self.y_est(features)
        dy = self.dy_est(features)
        ddy = self.ddy_est(features)
        p = tf.concat([x, y, dy, ddy], -1)

        return p
