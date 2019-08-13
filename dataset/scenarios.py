from math import pi

import numpy as np
import tensorflow as tf

from utils.utils import Pose2D


class Task:
    def __init__(self, x0, y0, th0, xk, yk, thk):
        self.x0 = x0
        self.y0 = y0
        self.th0 = th0
        self.xk = xk
        self.yk = yk
        self.thk = thk

    def get_definition(self):
        return self.x0, self.y0, self.th0, self.xk, self.yk, self.thk


def get_obstacles(path):
    with open(path, 'r') as fh:
        data = fh.read().split('\n')[1:-1]
    obs = []
    for line in data:
        o = np.array([float(x) for x in line.split()])
        o = np.reshape(o, (4, 2))
        obs.append(o)

    return tf.cast(tf.stack(obs, 0), tf.float32)


def _load_scenario(path: str) -> (Pose2D, Pose2D, tf.Tensor, tf.Tensor):
    with open(path, 'r') as fh:
        lines = fh.read().split('\n')[:-1]
    free_space = get_obstacles(lines[0])[tf.newaxis] # simulate batch
    map_matrix_path = lines[1]
    # distance_map = read_mat(map_matrix_path)
    distance_map = None
    start = Pose2D(*[float(x) for x in lines[2].split()])
    target = Pose2D(*[float(x) for x in lines[3].split()])
    return start, target, free_space, distance_map


def planning_dataset(path, num_of_samples):
    _, target, free_space, distance_map = _load_scenario(path)

    # skos
    #x0_range = tf.random_uniform([num_of_samples, 1], 48., 52.)
    ##x0_range = tf.random_uniform([num_of_samples, 1], 50., 50.)
    #y0_range = tf.random_uniform([num_of_samples, 1], 19., 23.)
    ##y0_range = tf.random_uniform([num_of_samples, 1], 20., 20.)
    #th0_range = tf.random_uniform([num_of_samples, 1], -0.15, 0.15) + pi/2
    ##th0_range = tf.random_uniform([num_of_samples, 1], -0.20, 0.20) + pi/2
    ##th0_range = tf.random_uniform([num_of_samples, 1], 0.0, 0.0) + pi/2

    # korytarz
    #x0_range = tf.random_uniform([num_of_samples, 1], 48.73, 48.77)
    x0_range = tf.random_uniform([num_of_samples, 1], 48.75, 48.75)
    #y0_range = tf.random_uniform([num_of_samples, 1], 11.18, 11.22)
    y0_range = tf.random_uniform([num_of_samples, 1], 11.2, 11.2)
    #th0_range = tf.random_uniform([num_of_samples, 1], -0.01, -0.11) + pi/2
    th0_range = tf.random_uniform([num_of_samples, 1], -0.05, -0.05) + pi/2

    # prost
    #x0_range = tf.random_uniform([num_of_samples, 1], 49., 51.)
    #y0_range = tf.random_uniform([num_of_samples, 1], 22., 24.)
    #th0_range = tf.random_uniform([num_of_samples, 1], 0.1, 0.1) + pi/2

    xk = tf.ones_like(x0_range) * target.x
    yk = tf.ones_like(x0_range) * target.y
    thk = tf.ones_like(x0_range) * target.fi

    ds = tf.data.Dataset.from_tensor_slices((x0_range, y0_range, th0_range, xk, yk, thk)) \
        .shuffle(num_of_samples)

    return ds, num_of_samples, free_space
