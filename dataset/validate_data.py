import os
import tensorflow as tf
from math import pi
from random import *
import numpy as np
import cv2

from utils.crucial_points import calculate_car_crucial_points
from utils.distances import dist

import matplotlib.pyplot as plt

tf.enable_eager_execution()

def _plot(x_path, y_path, th_path, data, step, n, print=False):
    _, _, free_space, _ = data
    for i in range(len(x_path)):
        x = x_path[i][n]
        y = y_path[i][n]
        th = th_path[i][n]
        cp = calculate_car_crucial_points(x, y, th)
        for p in cp:
            plt.plot(p[:, 0], p[:, 1], 'r*')

    for i in range(free_space.shape[1]):
        for j in range(4):
            fs = free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    #plt.xlim(-25.0, 25.0)
    #plt.ylim(0.0, 50.0)
    # plt.xlim(-15.0, 20.0)
    # plt.ylim(0.0, 35.0)
    #plt.xlim(-35.0, 5.0)
    #plt.ylim(-2.0, 6.0)
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()

def invalidate(x, y, fi, free_space):
    """
        Check how much specified points violate the environment constraints
    """
    crucial_points = calculate_car_crucial_points(x, y, fi)
    crucial_points = tf.stack(crucial_points, -2)

    penetration = dist(free_space, crucial_points)

    in_obstacle = tf.reduce_sum(penetration, -1)
    violation_level = tf.reduce_sum(in_obstacle, -1)

    # violation_level = integral(env.free_space, crucial_points)
    return violation_level

#path = "../../data/train/tunel/"
#path = "../../data/val/tunel/"
#path = "../../data/train/parkowanie_prostopadle/"
#path = "../../data/val/parkowanie_prostopadle/"
path = "../../TG_data/val/mix3/"

def read_scn(scn_path):
    scn_path = os.path.join(path, scn_path)
    data = np.loadtxt(scn_path, delimiter=' ', dtype=np.float32)
    map = tf.reshape(data, (3, 4, 2))[tf.newaxis]
    res_path = scn_path[:-3] + "scn"
    data = np.loadtxt(res_path, delimiter=' ', dtype=np.float32)
    data = tf.reshape(data, (-1, 2, 3))
    data = tf.concat([data[:, :, 1:], data[:, :, :1]], axis=-1)
    p0, pk = tf.unstack(data, axis=1)
    return p0, pk, map, scn_path


scenarios = [read_scn(f) for f in sorted(os.listdir(path)) if f.endswith(".map")]

for i, s in enumerate(scenarios):
    p0, pk, free_space, p = s

    x0, y0, fi0 = tf.unstack(p0[:, tf.newaxis], axis=-1)
    x1, y1, fi1 = tf.unstack(pk[:, tf.newaxis], axis=-1)

    #x0 = np.array(s[0][0])[:, np.newaxis]
    #y0 = np.array(s[0][1])[:, np.newaxis]
    #fi0 = np.array(s[0][2])[:, np.newaxis]

    #x1 = np.array([s[1][0] for s in scenarios])[:, np.newaxis]
    #y1 = np.array([s[1][1] for s in scenarios])[:, np.newaxis]
    #fi1 = np.array([s[1][2] for s in scenarios])[:, np.newaxis]
    #if i != 22: continue

    a0 = list(invalidate(x0, y0, fi0, free_space).numpy())
    a1 = list(invalidate(x1, y1, fi1, free_space).numpy())

    for k in range(len(a0)):
        if a0[k] > 0 or a1[k] > 0:
            print(p, k, " ", a0[k], " ", a1[k])
            _plot([x0, x1], [y0, y1], [fi0, fi1], s, i * 1000 + k, k)


