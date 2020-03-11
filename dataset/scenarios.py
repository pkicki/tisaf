import os
from random import shuffle

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()

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

    return tf.cast(tf.stack(rects, 0), tf.float32)




def planning_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
        map = tf.reshape(data, (3, 4, 2))
        res_path = scn_path[:-3] + "scn"
        data = np.loadtxt(res_path, delimiter=' ', dtype=np.float32)
        data = tf.reshape(data, (-1, 2, 3))
        data = tf.concat([data[:, :, 1:], data[:, :, :1]], axis=-1)
        p0, pk = tf.unstack(data, axis=1)
        return p0, pk, map

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".map")]

    maps = []
    p0 = []
    pk = []
    g = list(range(len(scenarios)))
    shuffle(g)
    for i in g:
        s = list(range(len(scenarios[i][0])))
        shuffle(s)
        for k in s:
            p0.append(scenarios[i][0][k])
            pk.append(scenarios[i][1][k])
            maps.append(scenarios[i][2])

    ds = tf.data.Dataset.from_tensor_slices((p0, pk, maps)) \
        .shuffle(buffer_size=len(maps), reshuffle_each_iteration=True)

    return ds, len(maps)


def planning(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
        map = tf.reshape(data, (3, 4, 2))
        res_path = scn_path[:-3] + "scn"
        data = np.loadtxt(res_path, delimiter=' ', dtype=np.float32)
        data = tf.reshape(data, (-1, 2, 3))
        data = tf.concat([data[:, :, 1:], data[:, :, :1]], axis=-1)
        p0, pk = tf.unstack(data, axis=1)
        return p0, pk, map

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".map")]

    print(sorted(os.listdir(path)))
    return scenarios


def planning_test(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
        map = tf.reshape(data, (3, 4, 2))
        res_path = scn_path[:-7] + "scenarios.scn"
        data = np.loadtxt(res_path, delimiter=' ', dtype=np.float32)
        data = tf.reshape(data, (-1, 2, 3))
        data = tf.concat([data[:, :, 1:], data[:, :, :1]], axis=-1)
        p0, pk = tf.unstack(data, axis=1)
        rrt_path = scn_path[:-7] + "rrt.txt"
        rrt = np.loadtxt(rrt_path, delimiter=' ', dtype=np.float32)
        sl_path = scn_path[:-7] + "sl.txt"
        sl = np.loadtxt(sl_path, delimiter=' ', dtype=np.float32)
        #map = tf.tile(map[tf.newaxis], (len(p0), 1, 1, 1))
        return p0, pk, map, rrt, sl

    scenarios = []
    for e in sorted(os.listdir(path)):
        for f in os.listdir(os.path.join(path, e)):
            for g in os.listdir(os.path.join(path, e, f)):
                if g.endswith("map"):
                    g = os.path.join(path, e, f, g)
                    scenarios.append(read_scn(g))

    return scenarios

def planning_tensor(path):
    def parse_function(scn_path):
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        # data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
        p0 = tf.unstack(data[0][:3], axis=0)
        pk = tf.unstack(data[-1][:3], axis=0)
        return p0, pk

    return [parse_function(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".scn")]


def load_map(path):
    data = np.loadtxt(path, delimiter=' ', dtype=np.float32)
    free_space = tf.transpose(tf.reshape(data, (3, 2, 4)), (0, 2, 1))
    return free_space


def decode_data(data):
    p0 = data[:, :1]
    pk = data[:, 1:]
    x0, y0, th0 = tf.unstack(p0, axis=-1)
    xk, yk, thk = tf.unstack(pk, axis=-1)
    return x0, y0, th0, xk, yk, thk
