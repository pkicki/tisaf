from math import pi

import tensorflow as tf

from utils.crucial_points import calculate_car_crucial_points
from utils.distances import dist
from utils.poly5 import curvature, params
from utils.utils import _calculate_length, Rot
from matplotlib import pyplot as plt

tf.enable_eager_execution()


class EstimatorLayer(tf.keras.Model):
    """
    Parameter estimator layer
    """

    def __init__(self, activation=tf.keras.activations.tanh, kernel_init_std=0.1, bias=0):
        super(EstimatorLayer, self).__init__()
        self.bias = bias
        self.features = [
            # tf.keras.layers.Dense(64, activation,
            #                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(1, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
        ]

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.features:
            x = layer(x)
        x += self.bias
        return x


class FeatureExtractorLayer(tf.keras.Model):
    """
    Feature exrtactor layer
    """

    def __init__(self, num_features, input_shape, activation=tf.keras.activations.tanh, kernel_init_std=0.1):
        super(FeatureExtractorLayer, self).__init__()
        self.features = [
            tf.keras.layers.Dense(32, activation,
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


class CorrectionLayer(tf.keras.Model):
    """
    Feature correction layer
    """

    def __init__(self, num_features, activation=tf.keras.activations.tanh, kernel_init_std=0.1, bias_init_val=0):
        super(CorrectionLayer, self).__init__()
        self.fc = tf.keras.layers.Dense(num_features, activation,
                                        kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std),
                                        bias_initializer=tf.keras.initializers.Constant(bias_init_val))

    def call(self, inputs, training=None):
        x = inputs
        x = self.fc(x)
        return x


class PlanningNetworkMP(tf.keras.Model):

    def __init__(self, num_segments, input_shape):
        super(PlanningNetworkMP, self).__init__()

        n = 256
        # n = 128

        self.preprocessing_stage = FeatureExtractorLayer(n, input_shape)
        self.processing_stage = [(EstimatorLayer(tf.exp, bias=1e-3),
                                  EstimatorLayer(),
                                  EstimatorLayer(),
                                  EstimatorLayer(),
                                  CorrectionLayer(n)) for _ in range(num_segments)]

    def call(self, task, training=None):
        x0, y0, th0, xk, yk, thk = task.get_definition()
        last_ddy = tf.zeros_like(x0)

        parameters = []
        for x_est, y_est, dy_est, ddy_est, corr_est in self.processing_stage:
            ex = (xk - x0) / 100.
            ey = (yk - y0) / 100.
            eth = (thk - th0) / (2 * pi)
            inputs = tf.concat([ex, ey, eth, last_ddy], -1)

            features = self.preprocessing_stage(inputs, training)

            x = x_est(features, training)
            y = y_est(features, training)
            dy = dy_est(features, training)
            ddy = ddy_est(features, training)
            p = tf.concat([x, y, dy, ddy], -1)
            parameters.append(p)

            x0, y0, th0 = calculate_next_point(p, x0[:, 0], y0[:, 0], th0[:, 0], last_ddy[:, 0])
            x0 = x0[:, tf.newaxis]
            y0 = y0[:, tf.newaxis]
            th0 = th0[:, tf.newaxis]
            last_ddy = ddy

        parameters = tf.stack(parameters, -1)

        return parameters


def calculate_next_point(plan, xL, yL, thL, last_ddy):
    x = plan[:, 0]
    tan_th = plan[:, 2]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

    return x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


class PlanningNetwork(tf.keras.Model):

    def __init__(self, num_segments, input_shape):
        super(PlanningNetwork, self).__init__()

        n = 256
        # n = 128

        self.preprocessing_stage = FeatureExtractorLayer(n, input_shape)
        self.processing_stage = [(EstimatorLayer(tf.exp, bias=8),
                                  EstimatorLayer(),
                                  EstimatorLayer(bias=-0.1),
                                  EstimatorLayer(),
                                  CorrectionLayer(n)) for _ in range(num_segments)]

    def call(self, task, training=None):
        x0, y0, th0, xk, yk, thk = task.get_definition()
        ex = (xk - x0) / 100.
        ey = (yk - y0) / 100.
        eth = (thk - th0) / (2 * pi)
        inputs = tf.concat([ex, ey, eth], -1)

        features = self.preprocessing_stage(inputs, training)

        parameters = []
        for x_est, y_est, dy_est, ddy_est, corr_est in self.processing_stage:
            x = x_est(features, training)
            y = y_est(features, training)
            dy = dy_est(features, training)
            ddy = ddy_est(features, training)
            features += corr_est(features, training)
            p = tf.concat([x, y, dy, ddy], -1)
            parameters.append(p)

        parameters = tf.stack(parameters, -1)

        return parameters


class Poly(tf.keras.Model):

    def __init__(self, num_segments, input_shape):
        super(Poly, self).__init__()
        self.x = tf.Variable(10.0, trainable=True)
        self.y = tf.Variable(-8.0, trainable=True)
        self.dy = tf.Variable(-0.5, trainable=True)
        self.ddy = tf.Variable(0.0, trainable=True)

        self.x1 = tf.Variable(10.0, trainable=True)
        self.y1 = tf.Variable(-8.0, trainable=True)
        self.dy1 = tf.Variable(-0.5, trainable=True)
        self.ddy1 = tf.Variable(0.0, trainable=True)

    def call(self, task, training=None):
        p = tf.stack([self.x, self.y, self.dy, self.ddy], -1)[tf.newaxis, :, tf.newaxis]
        p = tf.tile(p, [2, 1, 1])
        p1 = tf.stack([self.x1, self.y1, self.dy1, self.ddy1], -1)[tf.newaxis, :, tf.newaxis]
        p1 = tf.tile(p1, [2, 1, 1])
        p = tf.concat([p, p1], -1)
        return p


def plan_loss(plan, task, env):
    num_gpts = plan.shape[-1]
    x0, y0, th0, xk, yk, thk = task.get_definition()
    xL = x0[:, 0]
    yL = y0[:, 0]
    thL = th0[:, 0]
    last_ddy = tf.zeros_like(xL)
    curvature_loss = 0.0
    obstacles_loss = 0.0
    length_loss = 0.0
    x_path = []
    y_path = []
    th_path = []
    # regular path
    for i in range(num_gpts):
        # with tf.GradientTape() as tape:
        x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL = \
            process_segment(plan[:, :, i], xL, yL, thL, last_ddy, env)
        curvature_loss += curvature_violation
        obstacles_loss += invalid

        # grad = tape.gradient(curvature_violation, plan[:, :, i])
        # grad = tape.gradient(curvature_loss, curvature_violation)
        # print(grad)

        length_loss += length
        x_path.append(x_glob)
        y_path.append(y_glob)
        th_path.append(th_glob)

    # finishing segment
    xyL = tf.stack([xL, yL], -1)
    xyk = tf.stack([xk, yk], 1)
    R = Rot(-thL)
    xyk_L = tf.squeeze(R @ (xyk - xyL[:, :, tf.newaxis]), -1)
    thk_L = -(thL[:, tf.newaxis] - thk)
    x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL = \
        process_segment(tf.concat([xyk_L, tf.tan(thk_L), tf.zeros_like(thk_L)], -1), xL, yL, thL, last_ddy, env)
    curvature_loss += curvature_violation
    obstacles_loss += invalid
    length_loss += length
    x_path.append(x_glob)
    y_path.append(y_glob)
    th_path.append(th_glob)

    #loss = 1e-1 * curvature_loss + obstacles_loss
    loss = curvature_loss + obstacles_loss
    return loss, obstacles_loss, x_path, y_path, th_path


def _plot(x_path, y_path, th_path, env):
    for i in range(len(x_path)):
        x = x_path[i][0]
        y = y_path[i][0]
        th = th_path[i][0]
        cp = calculate_car_crucial_points(x, y, th)
        for p in cp:
            plt.plot(p[:, 0], p[:, 1])

    for i in range(env.free_space.shape[1]):
        for j in range(4):
            fs = env.free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    plt.show()


def process_segment(plan, xL, yL, thL, last_ddy, env):
    x = plan[:, 0]
    tan_th = plan[:, 2]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

    # calcualte length of segment
    length = _calculate_length(x_glob, y_glob)

    # calculate violations
    curvature_violation = tf.reduce_sum(tf.nn.relu(tf.abs(curvature) - env.max_curvature), -1)
    #curvature_violation = tf.reduce_sum(tf.abs(curvature), -1)
    #curvature_violation = tf.reduce_sum(tf.square(curvature), -1)
    invalid = invalidate(x_glob, y_glob, th_glob, env)

    return x_glob, y_glob, th_glob, curvature_violation, invalid, length, x_glob[:, -1], y_glob[:, -1], th_glob[:,
                                                                                                        -1]  # thL + tf.atan(
    # tan_th)


def invalidate(x, y, fi, env):
    """
        Check how much specified points violate the environment constraints
    """
    crucial_points = calculate_car_crucial_points(x, y, fi)
    crucial_points = tf.stack(crucial_points, -2)
    d = tf.sqrt(tf.reduce_sum((crucial_points[:, 1:] - crucial_points[:, :-1]) ** 2, -1))
    penetration = dist(env.free_space, crucial_points)

    in_obstacle = tf.reduce_sum(d * penetration[:, :-1], -1)
    violation_level = tf.reduce_sum(in_obstacle, -1)
    return violation_level


def _calculate_global_xyth_and_curvature(params, x, xL, yL, thL):
    x_local_sequence = tf.expand_dims(x, -1)
    x_local_sequence *= tf.linspace(0.0, 1.0, 128)
    curv, dX, dY = curvature(params, x_local_sequence)

    X = tf.stack([x_local_sequence ** 5, x_local_sequence ** 4, x_local_sequence ** 3, x_local_sequence ** 2,
                  x_local_sequence, tf.ones_like(x_local_sequence)], -1)
    y_local_sequence = tf.squeeze(X @ params, -1)
    R = Rot(thL)
    xy_glob = R @ tf.stack([x_local_sequence, y_local_sequence], 1)
    xy_glob += tf.expand_dims(tf.stack([xL, yL], -1), -1)

    x_glob, y_glob = tf.unstack(xy_glob, axis=1)

    th_glob = thL[:, tf.newaxis] + tf.atan(dY)
    return x_glob, y_glob, th_glob, curv
