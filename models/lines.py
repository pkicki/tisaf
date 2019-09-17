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
            # tf.keras.layers.Dense(64, activation,
            #                     kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
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
            #tf.keras.layers.Dense(num_features, activation,
            #                      kernel_initializer=tf.keras.initializers.RandomNormal(0.0, kernel_init_std)),
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

    def __init__(self, num_segments, input_shape):
        super(PlanningNetworkMP, self).__init__()

        #n = 256
        n = 128
        self.num_segments = num_segments - 1

        self.preprocessing_stage = FeatureExtractorLayer(n, input_shape, kernel_init_std=1.0)
        self.bn = tf.keras.layers.BatchNormalization()
        # self.x_est = EstimatorLayer(tf.nn.elu, bias=1.0, kernel_init_std=1.0)
        # self.x_est = EstimatorLayer(tf.abs, bias=1.0, kernel_init_std=1.0)
        self.x_est = EstimatorLayer(tf.nn.sigmoid, mul=10., bias=1.0, kernel_init_std=0.1, pre_bias=-1., pre_mul=1.0)
        self.th_est = EstimatorLayer(mul=pi/6., kernel_init_std=0.1)

    def call(self, data, training=None):
        x0, y0, th0, xk, yk, thk = decode_data(data)

        parameters = []
        for i in range(self.num_segments):
            ex = (xk - x0) / 2.5 - 1
            ey = (yk - y0) / 17.5 - 1
            eth = (thk - th0) / pi - 1
            inputs = tf.concat([ex, ey, eth], -1)

            features = self.preprocessing_stage(inputs, training)
            # print(features)

            x = self.x_est(features, training)
            th = self.th_est(features, training)
            p = tf.concat([x, th], -1)
            parameters.append(p)

            x0, y0, th0 = calculate_next_point(p, x0[:, 0], y0[:, 0], th0[:, 0])
            x0 = x0[:, tf.newaxis]
            y0 = y0[:, tf.newaxis]
            th0 = th0[:, tf.newaxis]

        parameters = tf.stack(parameters, -1)

        return parameters


def calculate_next_point(plan, xL, yL, thL):
    dx = plan[:, 0]
    dth = plan[:, 1]

    # calculate xy coords of segment
    x_glob, y_glob, th_glob = _calculate_global_xyth(dx, dth, xL, yL, thL)

    return x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


class Line(tf.keras.Model):

    def __init__(self, num_segments, input_shape):
        super(Line, self).__init__()
        self.x0 = tf.Variable(4.0, trainable=True, name="x0")
        self.th0 = tf.Variable(0.0, trainable=True, name="th0")

        self.x1 = tf.Variable(4.0, trainable=True, name="x1")
        self.th1 = tf.Variable(0.0, trainable=True, name="th1")

        self.x2 = tf.Variable(4.0, trainable=True, name="x2")
        self.th2 = tf.Variable(0.0, trainable=True, name="th2")

        self.x3 = tf.Variable(4.0, trainable=True, name="x3")
        self.th3 = tf.Variable(0.0, trainable=True, name="th3")

        self.x4 = tf.Variable(4.0, trainable=True, name="x4")
        self.th4 = tf.Variable(0.0, trainable=True, name="th4")

        self.x5 = tf.Variable(4.0, trainable=True, name="x5")
        self.th5 = tf.Variable(0.0, trainable=True, name="th5")

    def call(self, task, training=None):
        n = 1
        p0 = tf.stack([self.x0, self.th0], -1)[tf.newaxis, :, tf.newaxis]
        p0 = tf.tile(p0, [n, 1, 1])
        p1 = tf.stack([self.x1, self.th1], -1)[tf.newaxis, :, tf.newaxis]
        p1 = tf.tile(p1, [n, 1, 1])
        p2 = tf.stack([self.x2, self.th2], -1)[tf.newaxis, :, tf.newaxis]
        p2 = tf.tile(p2, [n, 1, 1])
        p3 = tf.stack([self.x3, self.th3], -1)[tf.newaxis, :, tf.newaxis]
        p3 = tf.tile(p3, [n, 1, 1])
        p4 = tf.stack([self.x4, self.th4], -1)[tf.newaxis, :, tf.newaxis]
        p4 = tf.tile(p4, [n, 1, 1])
        p5 = tf.stack([self.x5, self.th5], -1)[tf.newaxis, :, tf.newaxis]
        p5 = tf.tile(p5, [n, 1, 1])
        p = tf.concat([p0, p1, p2, p3, p4, p5], -1)
        return p


def plan_loss(plan, data, env):
    num_gpts = plan.shape[-1]
    x0, y0, th0, xk, yk, thk = decode_data(data)
    xL = x0[:, 0]
    yL = y0[:, 0]
    thL = th0[:, 0]
    curvature_loss = 0.0
    obstacles_loss = 0.0
    length_loss = 0.0
    x_path = []
    y_path = []
    th_path = []
    # regular path
    for i in range(num_gpts):
        # with tf.GradientTape() as tape:
        x_glob, y_glob, th_glob, invalid, length, xL, yL, thL = process_segment(plan[:, :, i], xL, yL, thL, env)
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
    xyk = tf.concat([xk, yk], 1)
    xyk_L = xyk - xyL
    dx = tf.sqrt(tf.reduce_sum(tf.square(xyk_L), -1))
    dth = tf.atan2(xyk_L[:, 1], xyk_L[:, 0]) - thL
    # overshoot_loss = tf.square(tf.nn.relu(xyL_k[:, 0])) + tf.nn.relu(tf.abs(thk_L) - pi / 2)
    # overshoot_loss = tf.nn.relu(xyL_k[:, 0]) + tf.nn.relu(tf.abs(thk_L) - pi / 2)
    x_glob, y_glob, th_glob, invalid, length, xL, yL, thL = \
        process_segment(tf.stack([dx, dth], -1), xL, yL, thL, env)
    obstacles_loss += invalid
    length_loss += length
    x_path.append(x_glob)
    y_path.append(y_glob)
    th_path.append(th_glob)

    # loss = 1e-1 * curvature_loss + obstacles_loss
    #loss = curvature_loss + obstacles_loss + overshoot_loss * 1e2
    # loss = obstacles_loss #+ overshoot_loss * 1e2
    # loss = overshoot_loss * 1e2
    loss = obstacles_loss
    return loss, obstacles_loss, x_path, y_path, th_path


def _plot(x_path, y_path, th_path, env, step):
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
    # plt.xlim(0.0, 15.0)
    # plt.ylim(0.0, 15.0)
    plt.xlim(-15.0, 20.0)
    plt.ylim(0.0, 35.0)
    plt.savefig("last_path" + str(step).zfill(6) + ".png")
    plt.clf()
    # plt.show()


def process_segment(plan, xL, yL, thL, env):
    dx = plan[:, 0]
    dth = plan[:, 1]

    # calculate xy coords of segment
    x_glob, y_glob, th_glob = _calculate_global_xyth(dx, dth, xL, yL, thL)

    # calcualte length of segment
    length, segments = _calculate_length(x_glob, y_glob)

    # calculate violations
    invalid = invalidate(x_glob, y_glob, th_glob, env)

    return x_glob, y_glob, th_glob, invalid, length, x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


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

    # violation_level = integral(env.free_space, crucial_points)
    return violation_level


def _calculate_global_xyth(dx, dth, xL, yL, thL):
    x_local_sequence = tf.expand_dims(dx, -1)
    x_local_sequence *= tf.linspace(0.0, 1.0, 128)

    y_local_sequence = 0.0 * x_local_sequence
    R = Rot(thL + dth)
    xy_glob = R @ tf.stack([x_local_sequence, y_local_sequence], 1)
    xy_glob += tf.expand_dims(tf.stack([xL, yL], -1), -1)

    x_glob, y_glob = tf.unstack(xy_glob, axis=1)

    th_glob = thL[:, tf.newaxis] + tf.tile(dth[:, tf.newaxis], [1, 128])
    return x_glob, y_glob, th_glob
