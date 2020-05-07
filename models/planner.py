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
    curvature_accumulation_loss = 0.0
    lengths = []
    x_path = []
    y_path = []
    th_path = []
    # regular path
    for i in range(num_gpts):
        #xL = x0
        #yL = y0
        #thL = th0
        x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL, curvature_sum = \
            process_segment(plan[:, :, i], xL, yL, thL, last_ddy, free_space, path)
        curvature_loss += curvature_violation
        obstacles_loss += invalid
        curvature_accumulation_loss += curvature_sum

        length_loss += length
        lengths.append(length)
        x_path.append(x_glob)
        y_path.append(y_glob)
        th_path.append(th_glob)
        dth = tf.atan(plan[:, 2, i])
        m = tf.cos(dth)**3
        last_ddy = plan[:, 3, i] * m

    ## finishing segment
    #xyL = tf.stack([xL, yL], -1)
    #xyk = tf.stack([xk, yk], 1)
    #R = Rot(-thL)
    #xyk_L = tf.squeeze(R @ (xyk - xyL)[:, :, tf.newaxis], -1)
    #xyL_k = tf.squeeze(Rot(-thk) @ (xyL - xyk)[:, :, tf.newaxis], -1)
    #thk_L = (thk - thL)[:, tf.newaxis]
    #overshoot_loss = tf.nn.relu(-xyk_L[:, 0]) + 1e2 * tf.nn.relu(tf.abs(thk_L[:, 0]) - pi / 2) + tf.nn.relu(xyL_k[:, 0])
    #x_glob, y_glob, th_glob, curvature_violation, invalid, length, xL, yL, thL, curvature_sum = \
    #    process_segment(tf.concat([xyk_L, tf.tan(thk_L), very_last_ddy], -1), xL, yL, thL, last_ddy, free_space, path)
    #curvature_loss += curvature_violation
    #obstacles_loss += invalid
    #curvature_accumulation_loss += curvature_sum
    #length_loss += length
    #lengths.append(length)
    #x_path.append(x_glob)
    #y_path.append(y_glob)
    #th_path.append(th_glob)

    lengths = tf.stack(lengths, -1)
    non_balanced_loss = tf.reduce_sum(
        tf.nn.relu(lengths - 1.5 * length_loss[:, tf.newaxis] / tf.cast(tf.shape(lengths)[-1], tf.float32)), -1)
    non_balanced_loss += tf.reduce_sum(
        tf.nn.relu(length_loss[:, tf.newaxis] / tf.cast(tf.shape(lengths)[-1], tf.float32) - lengths * 1.5), -1)

    dx = tf.nn.relu(tf.abs(xk - xL) - 0.4)
    dy = tf.nn.relu(tf.abs(yk - yL) - 0.4)
    dth = 10 * tf.nn.relu(tf.abs(thk - thL) - 0.1)
    overshoot_loss = dx + dy + dth

    #curvature_loss *= 3.
    # loss for pretraining
    #loss = non_balanced_loss + 1e2 * overshoot_loss + length_loss + 1e1 * curvature_loss
    # loss for training
    curvature_loss *= 1e1
    #coarse_loss = curvature_loss + obstacles_loss + overshoot_loss + non_balanced_loss
    #fine_loss = curvature_loss + obstacles_loss + overshoot_loss + non_balanced_loss + 1e-1 * curvature_accumulation_loss#length_loss
    #loss = tf.where(curvature_loss + obstacles_loss + overshoot_loss == 0, fine_loss, coarse_loss)
    #loss = coarse_loss
    loss = overshoot_loss

    return loss, obstacles_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path


def _plot(segments, data, step, print=False):
    _, _, free_space, path = data
    #plt.imshow(free_space[0])
    for i in range(len(segments)):
        x = segments[i][0][0]
        y = segments[i][1][0]
        th = segments[i][2][0]
        cp = calculate_car_crucial_points(x, y, th)
        for p in cp:
            plt.plot(p[:, 0], p[:, 1])
            #plt.plot(p[:, 0] * 10, (10. - p[:, 1]) * 10)

    for i in range(free_space.shape[1]):
        for j in range(4):
            fs = free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
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
    curvature_sum = tf.reduce_sum(tf.abs(curvature), -1)
    invalid = invalidate(x_glob, y_glob, th_glob, free_space, path)

    return x_glob, y_glob, th_glob, curvature_violation, invalid, length, x_glob[:, -1], y_glob[:, -1], th_glob[:, -1], curvature_sum


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

    #penetration = dist(free_space, crucial_points)
    #in_obstacle = tf.reduce_sum(d * penetration[:, :-1], -1)
    #violation_level = tf.reduce_sum(in_obstacle, -1)

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
