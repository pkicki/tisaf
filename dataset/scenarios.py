import os

import numpy as np
import tensorflow as tf
import matplotlib.image as mpimg
from utils.utils import Pose2D

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
    def parse_function(p0, pk, free_space, img_path):
        map_img = tf.image.decode_png(img_path, 1)
        map_img = tf.image.resize(map_img, (200, 200))
        map_img = tf.cast(tf.concat([tf.equal(map_img, 0), tf.not_equal(map_img, 0)], -1), tf.float32)
        return p0, pk, free_space, map_img

    def read_scn(scn_path):
        scn_path = os.path.join(path, scn_path)
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        p0 = tf.unstack(data[0][:4], axis=0)
        pk = tf.unstack(data[-1][:3], axis=0)
        return p0, pk

    def read_map(map_path):
        map_path = os.path.join(path, map_path)
        return get_map(map_path)

    scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".scn")]
    maps = [read_map(f) for f in sorted(os.listdir(path)) if f.endswith(".map")]
    imgs = [path + f for f in sorted(os.listdir(path)) if f.endswith(".png")]
    #for i, m in enumerate(maps):
    #    if not m.shape == (2, 4, 2):
    #        print("ZUO")
    #        print(m)
    #        print(sorted([f for f in os.listdir(path) if f.endswith(".map")])[i])

    def gen():
        for i in range(len(scenarios)):
            map_img = mpimg.imread(imgs[i])
            map_img = tf.cast(tf.stack([tf.equal(map_img, 0), tf.not_equal(map_img, 0)], -1), tf.float32)
            yield scenarios[i][0], scenarios[i][1], maps[i], map_img

    ds = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32, tf.float32, tf.float32)) \
        .shuffle(buffer_size=len(scenarios), reshuffle_each_iteration=True)
        #.map(parse_function, 8) \

    return ds, len(scenarios)


def planning_tensor(path):
    def parse_function(scn_path):
        data = np.loadtxt(scn_path, delimiter='\t', dtype=np.float32)
        # data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
        p0 = tf.unstack(data[0][:3], axis=0)
        pk = tf.unstack(data[-1][:3], axis=0)
        return p0, pk

    return [parse_function(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".scn")]


def load_map(path):
    map_path = os.path.join(path, "map.map")
    free_space = get_map(map_path)
    return free_space


def decode_data(data):
    p0 = data[:, :1]
    pk = data[:, 1:]
    x0, y0, th0 = tf.unstack(p0, axis=-1)
    xk, yk, thk = tf.unstack(pk, axis=-1)
    return x0, y0, th0, xk, yk, thk
