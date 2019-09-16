import os

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


def get_map(path):
    with open(path, 'r') as fh:
        data = fh.read().split('\n')[:-1]
    rects = []
    for line in data:
        o = np.array([float(x) for x in line.split()])
        o = np.reshape(o, (4, 2))
        rects.append(o)

    return tf.cast(tf.stack(rects, 0), tf.float32)[tf.newaxis]


def planning_dataset(path):
    def parse_function(scn_path):
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        #data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
        p0 = tf.unstack(data[0][:3], axis=0)
        pk = tf.unstack(data[-1][:3], axis=0)
        return p0, pk

    map_path = os.path.join(path, "map.map")
    scenarios = [parse_function(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".scn")]
    free_space = get_map(map_path)

    ds = tf.data.Dataset.from_tensor_slices(scenarios) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)

    return ds, len(scenarios), free_space


def decode_data(data):
    p0 = data[:, :1]
    pk = data[:, 1:]
    x0, y0, th0 = tf.unstack(p0, axis=-1)
    xk, yk, thk = tf.unstack(pk, axis=-1)
    return x0, y0, th0, xk, yk, thk
