from math import pi

import tensorflow as tf
import numpy as np

from utils.constants import Car
from utils.crucial_points import calculate_car_crucial_points
from utils.distances import dist, path_dist, if_inside, path_line_dist, path_dist_cp
from utils.poly5 import curvature, params, params_xy
from utils.utils import _calculate_length, Rot
from matplotlib import pyplot as plt

tf.enable_eager_execution()


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

        # self.map_processing = MapFeaturesProcessor(64)
        act = tf.keras.activations.tanh
        self.fc = [
            tf.keras.layers.Dense(256, activation=act),
            tf.keras.layers.Dense(256, activation=act),
        ]
        self.par = tf.keras.layers.Dense(37, activation=act)
        self.ms = tf.keras.layers.Dense(40)

    def call(self, data, map_features, training=None):
        p0, pk, free_space, path, xys = data
        R = np.array([25.6, 25.6, np.pi, 1.])[np.newaxis, np.newaxis]
        xys /= R

        # map_features = self.map_processing(free_space)
        bs, N, _ = tf.shape(xys)
        x = tf.reshape(xys, (bs, N * 4))
        for l in self.fc:
            x = l(x)
        ms = self.ms(x)
        ms = tf.reshape(ms, (bs, N - 1, 4))
        x = self.par(x)
        last_ddy = x[:, -1]
        x = x[:, :-1]
        dxys = tf.reshape(x, (bs, N-2, 4))
        dp0 = tf.zeros((bs, 1, 4))
        dpk = tf.concat([tf.zeros((bs, 1, 3)), last_ddy[:, tf.newaxis, tf.newaxis]], -1)
        dxys = tf.concat([dp0, dxys, dpk], 1)
        nxys = xys + dxys * 0.1
        nxys *= R
        #xd = np.array([0.1, 1.0, 0.1, 1.0])[np.newaxis, np.newaxis]
        #ms = tf.concat([xd, ms[:, 1:]], 1)
        #ms *= xd
        return nxys, ms


def plan_loss(nn_output, data):
    plan, ms = nn_output
    num_gpts = plan.shape[1]
    p0, pk, free_space, path, _ = data
    xk, yk, thk, _ = tf.unstack(pk, axis=-1)
    curvature_loss = 0.0
    obstacles_loss = 0.0
    length_loss = 0.0
    curvature_accumulation_loss = 0.0
    lengths = []
    x_path = []
    y_path = []
    th_path = []
    # regular path
    for i in range(num_gpts - 1):
        x_glob, y_glob, th_glob, curvature_violation, invalid, length, curvature_sum = \
            process_segment(plan[:, i:i + 2], free_space, path, ms[:, i])
        curvature_loss += curvature_violation
        obstacles_loss += invalid
        curvature_accumulation_loss += curvature_sum

        length_loss += length
        lengths.append(length)
        x_path.append(x_glob)
        y_path.append(y_glob)
        th_path.append(th_glob)

    dx = tf.nn.relu(tf.abs(xk - x_path[-1][:, -1]) - 0.4)
    dy = tf.nn.relu(tf.abs(yk - y_path[-1][:, -1]) - 0.4)
    dth = 10 * tf.nn.relu(tf.abs(thk - th_path[-1][:, -1]) - 0.1)
    overshoot_loss = dx + dy + dth

    # curvature_loss *= 3.
    # loss for pretraining
    # loss = non_balanced_loss + 1e2 * overshoot_loss + length_loss + 1e1 * curvature_loss
    # loss for training
    curvature_loss *= 1e1
    coarse_loss = curvature_loss + obstacles_loss + overshoot_loss
    fine_loss = curvature_loss + obstacles_loss + overshoot_loss + 1e-1 * curvature_accumulation_loss  # length_loss
    loss = tf.where(curvature_loss + obstacles_loss + overshoot_loss == 0, fine_loss, coarse_loss)
    # loss = coarse_loss

    return loss, obstacles_loss, overshoot_loss, curvature_loss, x_path, y_path, th_path


def _plot(x_path, y_path, th_path, data, step, print=False):
    _, _, free_space, path, xys = data
    # plt.imshow(free_space[0])
    for i in range(len(x_path)):
        x = x_path[i][0]
        y = y_path[i][0]
        th = th_path[i][0]
        cp = calculate_car_crucial_points(x, y, th)
        for p in cp:
            plt.plot(p[:, 0], p[:, 1])
            # plt.plot(p[:, 0] * 10, (10. - p[:, 1]) * 10)
    # plt.plot(path[0, :, 0] * 10, (10 - path[0, :, 1]) * 10, 'r')
    plt.plot(path[0, :, 0], path[0, :, 1], 'r')

    for i in range(free_space.shape[1]):
        for j in range(4):
            fs = free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    # plt.xlim(-25.0, 25.0)
    # plt.ylim(0.0, 50.0)
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


def process_segment(plan, free_space, path, ms):
    p = params_xy(plan[:, 0], plan[:, 1], ms)

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p)

    # calcualte length of segment
    length, segments = _calculate_length(x_glob, y_glob)

    # calculate violations
    overcurved = tf.nn.relu(tf.abs(curvature[:, 1:]) - Car.max_curvature)
    curvature_violation = tf.reduce_sum(overcurved * segments, -1)
    curvature_sum = tf.reduce_sum(tf.abs(curvature), -1)
    invalid = invalidate(x_glob, y_glob, th_glob, free_space, path)

    return x_glob, y_glob, th_glob, curvature_violation, invalid, length, curvature_sum


def invalidate(x, y, fi, free_space, path):
    """
        Check how much specified points violate the environment constraints
    """
    crucial_points = calculate_car_crucial_points(x, y, fi)
    crucial_points = tf.stack(crucial_points, -2)
    xy = tf.stack([x, y], axis=-1)[:, :, tf.newaxis]

    d = tf.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)

    path_cp = calculate_car_crucial_points(path[..., 0], path[..., 1], path[..., 2])
    path_cp = tf.stack(path_cp, -2)
    penetration = path_dist_cp(path_cp, crucial_points)
    not_in_collision = if_inside(free_space, crucial_points)
    not_in_collision = tf.reduce_any(not_in_collision, axis=-1)
    penetration = tf.where(not_in_collision, tf.zeros_like(penetration), penetration)
    violation_level = tf.reduce_sum(d[..., 0] * penetration[:, :-1], -1)

    # penetration = dist(free_space, crucial_points)
    # in_obstacle = tf.reduce_sum(d * penetration[:, :-1], -1)
    # violation_level = tf.reduce_sum(in_obstacle, -1)

    return violation_level


def _calculate_global_xyth_and_curvature(params):
    s = tf.linspace(0.0, 1.0, 128)
    curv, dx, dy, x, y = curvature(params, s)
    th = tf.atan2(dy, dx)
    #plt.plot(th.numpy()[0])
    #plt.plot(x.numpy()[0], y.numpy()[0])
    #plt.show()
    return x, y, th, curv
