from math import pi

import tensorflow as tf

from dataset.scenarios import decode_data
from utils.constants import Car
from utils.crucial_points import calculate_car_crucial_points
from utils.distances import dist, integral
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

    def call(self, inputs, training=None):
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

    def __init__(self, num_features, input_shape, activation=tf.keras.activations.tanh, kernel_init_std=0.1):
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

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.features:
            x = layer(x)
        # x = self.fc(x)
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


class PlanningNetworkMP(tf.keras.Model):

    def __init__(self, num_segments, input_shape):
        super(PlanningNetworkMP, self).__init__()

        n = 256
        self.num_segments = num_segments - 1

        self.map_processing = MapFeaturesProcessor(64)

        self.preprocessing_stage = FeatureExtractorLayer(n, input_shape)
        self.x_est = EstimatorLayer(tf.nn.sigmoid, mul=10., bias=0.1, kernel_init_std=0.1, pre_bias=-1., pre_mul=1.0)
        self.y_est = EstimatorLayer(tf.identity)
        self.dy_est = EstimatorLayer(tf.identity)
        self.ddy_est = EstimatorLayer(tf.identity)
        self.last_ddy_est = EstimatorLayer(tf.identity)

    def call(self, data, map_features, training=None):
        p0, pk, free_space = data
        x0, y0, th0 = tf.unstack(p0, axis=-1)
        ddy0 = tf.zeros_like(x0)
        xk, yk, thk = tf.unstack(pk, axis=-1)

        last_ddy = ddy0
        W = 20.
        H = 20.

        map_features = self.map_processing(free_space)

        parameters = []
        features = None
        for i in range(self.num_segments):
            inputs = tf.stack([x0 / W, y0 / H, th0 / (2 * pi), last_ddy, xk / W, yk / H, thk / (2 * pi)], -1)

            features = self.preprocessing_stage(inputs, training)
            features = tf.concat([features, map_features], -1)

            x = self.x_est(features, training)
            y = self.y_est(features, training)
            dy = self.dy_est(features, training)
            ddy = self.ddy_est(features, training)
            p = tf.concat([x, y, dy, ddy], -1)
            parameters.append(p)

            x0, y0, th0 = calculate_next_point(p, x0, y0, th0, last_ddy)
            last_ddy = ddy[:, 0]

        parameters = tf.stack(parameters, -1)

        last_ddy = self.last_ddy_est(features, training)

        return parameters, last_ddy


def calculate_next_point(plan, xL, yL, thL, last_ddy):
    x = plan[:, 0]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

    return x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


def plan_loss(plan, very_last_ddy, data):
    num_gpts = plan.shape[-1]
    p0, pk, free_space, path = data
    x0, y0, th0 = tf.unstack(p0, axis=-1)
    ddy0 = tf.zeros_like(x0)
    xk, yk, thk = tf.unstack(pk, axis=-1)
    xL = x0
    yL = y0
    thL = th0
    last_ddy = ddy0
    curvature_loss = 0.0
    obstacles_loss = 0.0
    length_loss = 0.0
    lengths = []
    x_path = []
    y_path = []
    th_path = []
    # regular path
    for i in range(num_gpts):
        #xL = x0
        #yL = y0
        #thL = th0
        x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL = \
            process_segment(plan[:, :, i], xL, yL, thL, last_ddy, free_space, path)
        curvature_loss += curvature_violation
        obstacles_loss += invalid
        print(i, invalid)

        length_loss += length
        lengths.append(length)
        x_path.append(x_glob)
        y_path.append(y_glob)
        th_path.append(th_glob)
        dth = tf.atan(plan[:, 2, i])
        m = tf.cos(dth)**3
        last_ddy = plan[:, 3, i] * m

    # finishing segment
    xyL = tf.stack([xL, yL], -1)
    xyk = tf.stack([xk, yk], 1)
    R = Rot(-thL)
    xyk_L = tf.squeeze(R @ (xyk - xyL)[:, :, tf.newaxis], -1)
    xyL_k = tf.squeeze(Rot(-thk) @ (xyL - xyk)[:, :, tf.newaxis], -1)
    thk_L = (thk - thL)[:, tf.newaxis]
    overshoot_loss = tf.nn.relu(-xyk_L[:, 0]) + 1e2 * tf.nn.relu(tf.abs(thk_L[:, 0]) - pi / 2) + tf.nn.relu(xyL_k[:, 0])
    x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL = \
        process_segment(tf.concat([xyk_L, tf.tan(thk_L), very_last_ddy], -1), xL, yL, thL, last_ddy, free_space, path)
    curvature_loss += curvature_violation
    obstacles_loss += invalid
    length_loss += length
    lengths.append(length)
    x_path.append(x_glob)
    y_path.append(y_glob)
    th_path.append(th_glob)

    lengths = tf.stack(lengths, -1)
    non_balanced_loss = tf.reduce_sum(
        tf.nn.relu(lengths - 1.5 * length_loss[:, tf.newaxis] / tf.cast(tf.shape(lengths)[-1], tf.float32)), -1)
    non_balanced_loss += tf.reduce_sum(
        tf.nn.relu(length_loss[:, tf.newaxis] / tf.cast(tf.shape(lengths)[-1], tf.float32) - lengths * 1.5), -1)

    # loss for pretraining
    #loss = non_balanced_loss + 1e2 * overshoot_loss + length_loss + curvature_loss
    # loss for training
    coarse_loss = curvature_loss + obstacles_loss + overshoot_loss + non_balanced_loss
    fine_loss = curvature_loss + obstacles_loss + overshoot_loss + non_balanced_loss + length_loss
    loss = tf.where(curvature_loss + obstacles_loss == 0, fine_loss, coarse_loss)

    return loss, obstacles_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path


def _plot(x_path, y_path, th_path, data, step, print=False):
    _, _, free_space, path = data
    plt.imshow(free_space)
    for i in range(len(x_path)):
        x = x_path[i][0]
        y = y_path[i][0]
        th = th_path[i][0]
        cp = calculate_car_crucial_points(x, y, th)
        for p in cp:
            plt.plot(p[:, 0], p[:, 1])

    #for i in range(free_space.shape[1]):
    #    for j in range(4):
    #        fs = free_space
    #        plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    #plt.xlim(-25.0, 25.0)
    #plt.ylim(0.0, 50.0)
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


def process_segment(plan, xL, yL, thL, last_ddy, free_space, path):
    x = plan[:, 0]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

    # calcualte length of segment
    length, segments = _calculate_length(x_glob, y_glob)

    # calculate violations
    curvature_violation = tf.reduce_sum(tf.nn.relu(tf.abs(curvature[:, 1:]) - Car.max_curvature) * segments, -1)
    invalid = invalidate(x_glob, y_glob, th_glob, free_space, path)

    return x_glob, y_glob, th_glob, curvature_violation, invalid, length, x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


def invalidate(x, y, fi, free_space, path):
    """
        Check how much specified points violate the environment constraints
    """
    crucial_points = calculate_car_crucial_points(x, y, fi)
    crucial_points = tf.stack(crucial_points, -2)
    xy = tf.stack([x, y], axis=-1)[:, :, tf.newaxis]

    d = tf.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)
    penetration = dist(path, xy)
    fi = tf.Variable([[[True]]], dtype=tf.bool)
    fi = tf.concat([tf.logical_not(fi), tf.tile(fi, (1, 127, 1))], 1)
    penetration = tf.where(fi, penetration, tf.zeros_like(penetration))

    in_obstacle = tf.reduce_sum(d * penetration[:, :-1], -1)
    #in_obstacle = tf.reduce_sum(d * penetration[:, :-1], -1)
    violation_level = tf.reduce_sum(in_obstacle, -1)

    return violation_level


def _calculate_global_xyth_and_curvature(params, x, xL, yL, thL):
    x_local_sequence = tf.expand_dims(x, -1)
    #x_local_sequence *= tf.linspace(0.0, 1.0, 64)
    x_local_sequence *= tf.linspace(0.0, 1.0, 128)
    curv, dX, dY = curvature(params, x_local_sequence)

    X = tf.stack([x_local_sequence ** 5, x_local_sequence ** 4, x_local_sequence ** 3, x_local_sequence ** 2,
                  x_local_sequence, tf.ones_like(x_local_sequence)], -1)
    y_local_sequence = tf.squeeze(X @ params, -1)
    R = Rot(thL)
    xy_glob = R @ tf.stack([x_local_sequence, y_local_sequence], 1)
    xyL = tf.stack([xL, yL], -1)[..., tf.newaxis]
    #xy_glob += tf.expand_dims(tf.stack([xL, yL], -1), -1)
    #xy_glob += 1e-10
    #xy_glob += 1e-6
    xy_glob += tf.constant(xyL, dtype=tf.float32)

    x_glob, y_glob = tf.unstack(xy_glob, axis=1)

    th_glob = thL[:, tf.newaxis] + tf.atan(dY)
    return x_glob, y_glob, th_glob, curv
