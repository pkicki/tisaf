from math import pi

import tensorflow as tf

from dataset.scenarios import decode_data
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
            tf.keras.layers.Dense(64, activation,
                                  kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
            tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
        ]

    def call(self, inputs, training=None):
        x = inputs
        for layer in self.features:
            x = layer(x)
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


class PlanningNetworkMP(tf.keras.Model):

    def __init__(self, input_shape):
        super(PlanningNetworkMP, self).__init__()

        n = 256

        self.preprocessing_stage = FeatureExtractorLayer(n, input_shape, kernel_init_std=1.0)
        # self.x_est = EstimatorLayer(tf.nn.elu, bias=1.0, kernel_init_std=1.0)
        # self.x_est = EstimatorLayer(tf.abs, bias=1.0, kernel_init_std=1.0)
        self.x_est = EstimatorLayer(tf.nn.sigmoid, mul=5., bias=0.0, kernel_init_std=0.1, pre_bias=-1., pre_mul=1.0)
        self.y_est = EstimatorLayer(tf.nn.sigmoid, mul=35., bias=0.0, kernel_init_std=0.1, pre_bias=-1., pre_mul=1.0)
        # self.th_est = EstimatorLayer(mul=pi/6., kernel_init_std=0.1)

    def call(self, x, y, xk, yk, training=None):
        ex = (xk - x) / 15.
        ey = (yk - y) / 15.
        inputs = tf.stack([ex, ey], -1)

        features = self.preprocessing_stage(inputs, training)
        # print(features)

        x = self.x_est(features, training)
        # th = self.th_est(features, training)
        y = self.y_est(features, training)
        p = tf.concat([x, y], -1)

        return p


def calculate_next_point(plan, xL, yL, thL):
    dx = plan[:, 0]
    dy = plan[:, 1]

    # calculate xy coords of segment
    x_glob, y_glob, th_glob = _calculate_global_xyth(dx, dy, xL, yL, thL)

    return x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


def plan_loss(plan, env, x0, y0, th0, xk, yk, thk):
    num_gpts = plan.shape[-1]
    xL = x0
    yL = y0
    thL = th0
    last_ddy = tf.zeros_like(xL)
    curvature_loss = 0.0
    obstacles_loss = 0.0
    length_loss = 0.0
    x_path = []
    y_path = []
    th_path = []
    lengths = []
    # regular path
    for i in range(num_gpts):
        # with tf.GradientTape() as tape:
        x_glob, y_glob, th_glob, invalid, length, dth_loss, xL, yL, thL = process_segment(plan[:, :, i], xL, yL, thL, env)
        obstacles_loss += invalid
        curvature_loss += dth_loss
        lengths.append(length)

        # grad = tape.gradient(curvature_violation, plan[:, :, i])
        # grad = tape.gradient(curvature_loss, curvature_violation)
        # print(grad)

        length_loss += length
        x_path.append(x_glob)
        y_path.append(y_glob)
        th_path.append(th_glob)

    # finishing segment
    dx = xk
    dy = yk
    # overshoot_loss = tf.square(tf.nn.relu(xyL_k[:, 0])) + tf.nn.relu(tf.abs(thk_L) - pi / 2)
    # overshoot_loss = tf.nn.relu(xyL_k[:, 0]) + tf.nn.relu(tf.abs(thk_L) - pi / 2)
    some_loss = tf.reduce_sum(tf.nn.relu(tf.abs(thk - thL) - pi / 6))
    x_glob, y_glob, th_glob, invalid, length, dth_loss, xL, yL, thL = \
        process_segment(tf.stack([dx, dy], -1), xL, yL, thL, env)
    #obstacles_loss += invalid
    #curvature_loss += dth_loss
    #length_loss += length
    x_path.append(x_glob)
    y_path.append(y_glob)
    th_path.append(th_glob)
    lengths.append(length)

    can_finish = tf.equal(invalid, 0.0).numpy()[0]

    # loss = 1e-1 * curvature_loss + obstacles_loss
    # loss = 1e-1 * curvature_loss + obstacles_loss + overshoot_loss * 1e0
    loss = obstacles_loss# + overshoot_loss * 1e0
    # loss = obstacles_loss #+ overshoot_loss * 1e2
    # loss = overshoot_loss * 1e2
    return loss, obstacles_loss, invalid, curvature_loss, x_path, y_path, th_path, can_finish


def _plot(x_path, y_path, th_path, env, step, plot=False):
    for i in range(len(x_path)):
        x = x_path[i][0]
        y = y_path[i][0]
        th = th_path[i][0]
        # cp = calculate_car_crucial_points(x, y, th)
        # for p in cp:
        # plt.plot(p[:, 0], p[:, 1])
        plt.plot(x, y)

    for i in range(env.free_space.shape[1]):
        for j in range(4):
            fs = env.free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    # plt.xlim(0.0, 15.0)
    # plt.ylim(0.0, 15.0)
    plt.xlim(-15.0, 20.0)
    plt.ylim(0.0, 35.0)
    if plot:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


def process_segment(plan, xL, yL, thL, env):
    dx = plan[:, 0]
    dy = plan[:, 1]

    # calculate xy coords of segment
    x_glob, y_glob, th_glob = _calculate_global_xyth(dx, dy, xL, yL, thL)

    dth_loss = tf.nn.relu(tf.abs(th_glob[:, -1] - thL) - pi / 6)

    # calcualte length of segment
    # length, segments = _calculate_length(x_glob, y_glob)
    length = tf.sqrt((dy - yL) ** 2 + (dx - xL) ** 2)

    # calculate violations
    invalid = invalidate(x_glob, y_glob, th_glob, env)

    return x_glob, y_glob, th_glob, invalid, length, dth_loss, x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


def invalidate(x, y, fi, env):
    """
        Check how much specified points violate the environment constraints
    """
    # crucial_points = calculate_car_crucial_points(x, y, fi)
    # crucial_points = tf.stack(crucial_points, -2)
    crucial_points = tf.stack([x, y], -1)[:, :, tf.newaxis]

    d = tf.sqrt(tf.reduce_sum((crucial_points[:, 1:] - crucial_points[:, :-1]) ** 2, -1))
    penetration = dist(env.free_space, crucial_points)

    in_obstacle = tf.reduce_sum(d * penetration[:, :-1], -1)
    violation_level = tf.reduce_mean(in_obstacle, -1)

    # violation_level = integral(env.free_space, crucial_points)
    return violation_level


def _calculate_global_xyth(dx, dy, xL, yL, thL):
    x_local_sequence = tf.expand_dims(dx - xL, -1)
    x_local_sequence *= tf.linspace(0.0, 1.0, 128)

    y_local_sequence = tf.expand_dims(dy - yL, -1)
    y_local_sequence *= tf.linspace(0.0, 1.0, 128)

    xy_glob = tf.stack([x_local_sequence, y_local_sequence], 1)
    xy_glob += tf.expand_dims(tf.stack([xL, yL], -1), -1)

    x_glob, y_glob = tf.unstack(xy_glob, axis=1)

    dth = tf.atan2(dy, dx)
    th_glob = tf.tile(dth[:, tf.newaxis], [1, 128])
    return x_glob, y_glob, th_glob
