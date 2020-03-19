import os
from random import shuffle

import numpy as np
import tensorflow as tf

tf.enable_eager_execution()


def planning_dataset(path):
    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
        map = tf.reshape(data, (3, 4, 2))
        res_path = scn_path[:-3] + "scn"
        p0 = []
        pk = []
        paths = []
        with open(res_path, 'r') as fh:
            lines = fh.read().split('\n')[:-1]
            for i, l in enumerate(lines):
                v = [float(x) for x in l.split()]
                if i % 3 == 0:
                    p0.append(v)
                elif i % 3 == 1:
                    pk.append(v)
                elif i % 3 == 2:
                    w = 702 - len(v)
                    v = np.pad(v, (0, w))
                    paths.append(v)
        p0 = np.array(p0, dtype=np.float32)
        p0 = np.concatenate([p0[:, 1:], p0[:, :1]], -1)
        pk = np.array(pk, dtype=np.float32)
        pk = np.concatenate([pk[:, 1:], pk[:, :1]], -1)
        paths = np.stack(paths, 0).astype(np.float32)
        paths = np.reshape(paths, (paths.shape[0], -1, 3))
        paths = paths[:, :, 1:]
        #paths = tf.concat([paths[:, :, 1:], paths[:, :, :1]], -1)
        return p0, pk, map, paths

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".map")]

    maps = []
    p0 = []
    pk = []
    paths = []
    g = list(range(len(scenarios)))
    shuffle(g)
    for i in g:
        s = list(range(len(scenarios[i][0])))
        shuffle(s)
        for k in s:
            p0.append(scenarios[i][0][k])
            pk.append(scenarios[i][1][k])
            paths.append(scenarios[i][3][k])
            maps.append(scenarios[i][2])

    ds = tf.data.Dataset.from_tensor_slices((p0, pk, maps, paths)) \
        .shuffle(buffer_size=len(maps), reshuffle_each_iteration=True)

    return ds, len(maps)


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

