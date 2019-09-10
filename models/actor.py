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
        self.x_est = EstimatorLayer(tf.nn.sigmoid, mul=10., bias=1.0, kernel_init_std=0.1, pre_bias=-5., pre_mul=1.0)
        self.y_est = EstimatorLayer(mul=10., pre_mul=0.1)
        self.dy_est = EstimatorLayer(mul=1., bias=0.0, pre_mul=0.1)
        self.ddy_est = EstimatorLayer(mul=2., pre_mul=0.1)

    def call(self, x, y, th, ddy, xk, yk, thk, training=None):
        ex = (xk - x) / 15.
        ey = (yk - y) / 15.
        eth = (thk - th) / (2 * pi)
        inputs = tf.stack([ex, ey, eth, ddy], -1)

        features = self.preprocessing_stage(inputs, training)

        x = self.x_est(features, training)
        y = self.y_est(features, training)
        dy = self.dy_est(features, training)
        ddy = self.ddy_est(features, training)
        p = tf.concat([x, y, dy, ddy], -1)

        return p


def calculate_next_point(plan, xL, yL, thL, last_ddy):
    x = plan[:, 0]
    tan_th = plan[:, 2]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

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
    xyk_L = tf.squeeze(R @ (xyk - xyL)[:, :, tf.newaxis], -1)
    xyL_k = tf.squeeze(Rot(-thk) @ (xyL - xyk)[:, :, tf.newaxis], -1)
    thk_L = (thk - thL)
    overshoot_loss = tf.nn.relu(-xyk_L[:, 0]) + 1e1 * tf.nn.relu(tf.abs(thk_L) - pi / 2) + tf.nn.relu(xyL_k[:, 0])
    # overshoot_loss = tf.square(tf.nn.relu(xyL_k[:, 0])) + tf.nn.relu(tf.abs(thk_L) - pi / 2)
    # overshoot_loss = tf.nn.relu(xyL_k[:, 0]) + tf.nn.relu(tf.abs(thk_L) - pi / 2)
    x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL = \
        process_segment(tf.concat([xyk_L, tf.tan(thk_L[:, tf.newaxis]), tf.zeros_like(thk_L)[:, tf.newaxis]], -1), xL, yL, thL, last_ddy, env)
    x_path.append(x_glob)
    y_path.append(y_glob)
    th_path.append(th_glob)

    can_finish = tf.equal(curvature_violation + invalid, 0.0).numpy()[0]

    # loss = 1e-1 * curvature_loss + obstacles_loss
    loss = 1e-1 * curvature_loss + obstacles_loss + overshoot_loss * 1e0
    # loss = obstacles_loss #+ overshoot_loss * 1e2
    # loss = overshoot_loss * 1e2
    return loss, obstacles_loss, overshoot_loss, curvature_loss, x_path, y_path, th_path, can_finish


def _plot(x_path, y_path, th_path, env, step, plot=False):
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
    if plot:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


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
    #curvature_violation = tf.reduce_sum(tf.nn.relu(tf.abs(curvature) - env.max_curvature), -1)
    curvature_violation = tf.reduce_mean(tf.nn.relu(tf.abs(curvature) - env.max_curvature), -1)
    # curvature_violation = tf.reduce_sum(tf.abs(curvature), -1)
    # curvature_violation = tf.reduce_sum(tf.square(curvature), -1)
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
    #violation_level = tf.reduce_sum(in_obstacle, -1)
    violation_level = tf.reduce_mean(in_obstacle, -1)

    # violation_level = integral(env.free_space, crucial_points)
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
