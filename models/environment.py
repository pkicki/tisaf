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


class Environment(object):
    def __init__(self):
        super(Environment, self).__init__()
        self._state = None

    def set_state(self, state):
        self._state = state

    def get_state(self):
        return self._state

    def _get_pose(self):
        p0, pk, free_space, path = self._state
        return tf.unstack(p0, axis=-1)

    def _get_free_space(self):
        p0, pk, free_space, path = self._state
        return free_space

    def step(self, action):
        # calculate xy coords of segment
        x_glob, y_glob, th_glob, curvature = self._calculate_global_xyth_and_curvature(action)

        loss, valid, terminal = self.loss(x_glob, y_glob, th_glob, curvature)
        reward = -loss

        path = (x_glob, y_glob, th_glob)
        dth = tf.atan(action[0, 2])
        m = tf.cos(dth) ** 3
        new_ddy = action[0, 3] * m

        new_state = [x_glob[0, -1], y_glob[0, -1], th_glob[0, -1], new_ddy]
        self.set_state((np.array(new_state)[np.newaxis], *self._state[1:]))
        return new_state, reward, terminal, valid, path

    def loss(self, x, y, th, curvature):
        p0, pk, free_space, path = self._state
        xk, yk, thk = tf.unstack(pk, axis=-1)

        dx = tf.nn.relu(tf.abs(xk - x[:, -1]) - 0.4) / 40.
        dy = tf.nn.relu(tf.abs(yk - y[:, -1]) - 0.4) / 40.
        dth = 10 * tf.nn.relu(tf.abs(thk - th[:, -1]) - 0.1) / (2 * np.pi)
        overshoot_loss = (dx + dy + dth) / 3.

        # calcualte length of segment
        length, segments = _calculate_length(x, y)

        # calculate violations
        curvature_loss = tf.reduce_sum(tf.nn.relu(tf.abs(curvature[:, 1:]) - Car.max_curvature) * segments, -1)
        curvature_accumulation_loss = tf.reduce_sum(tf.abs(curvature), -1)
        obstacles_loss = self._invalidate(x, y, th)

        curvature_loss *= 1e1
        # coarse_loss = curvature_loss + obstacles_loss + overshoot_loss
        # fine_loss = curvature_loss + obstacles_loss + overshoot_loss + 1e-1 * curvature_accumulation_loss
        # loss = tf.where(curvature_loss + obstacles_loss + overshoot_loss == 0, fine_loss, coarse_loss)
        # loss = coarse_loss
        #loss = curvature_loss + obstacles_loss + overshoot_loss + 1e-3 * curvature_accumulation_loss
        loss = overshoot_loss

        #valid = (curvature_loss + obstacles_loss == 0.)
        valid = (overshoot_loss == 0.)
        terminal = tf.cast(overshoot_loss == 0., tf.float32)

        return loss, valid, terminal

    def _calculate_global_xyth_and_curvature(self, action):
        xL, yL, thL, last_ddy = self._get_pose()
        # calculate params
        zeros = tf.zeros_like(last_ddy)
        p = params([zeros, zeros, zeros, last_ddy], tf.unstack(action, axis=1))

        x = action[:, 0]

        x_local_sequence = tf.expand_dims(x, -1)
        # x_local_sequence *= tf.linspace(0.0, 1.0, 64)
        x_local_sequence *= tf.linspace(0.0, 1.0, 128)
        curv, dX, dY = curvature(p, x_local_sequence)

        X = tf.stack([x_local_sequence ** 5, x_local_sequence ** 4, x_local_sequence ** 3, x_local_sequence ** 2,
                      x_local_sequence, tf.ones_like(x_local_sequence)], -1)
        y_local_sequence = tf.squeeze(X @ p, -1)
        R = Rot(thL)
        xy_glob = R @ tf.stack([x_local_sequence, y_local_sequence], 1)
        xyL = tf.stack([xL, yL], -1)[..., tf.newaxis]
        xy_glob += tf.constant(xyL, dtype=tf.float32)

        x_glob, y_glob = tf.unstack(xy_glob, axis=1)

        th_glob = thL[:, tf.newaxis] + tf.atan(dY)
        return x_glob, y_glob, th_glob, curv

    def _invalidate(self, x, y, fi):
        """
            Check how much specified points violate the environment constraints
        """
        free_space = self._get_free_space()
        crucial_points = calculate_car_crucial_points(x, y, fi)
        crucial_points = tf.stack(crucial_points, -2)
        xy = tf.stack([x, y], axis=-1)[:, :, tf.newaxis]

        d = tf.linalg.norm(xy[:, 1:] - xy[:, :-1], axis=-1)

        penetration = dist(free_space, crucial_points)
        in_obstacle = tf.reduce_sum(d * penetration[:, :-1], -1)
        violation_level = tf.reduce_sum(in_obstacle, -1)

        return violation_level
