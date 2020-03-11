import inspect
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import plan_loss, _plot, PlanningNetworkMP

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from utils.execution import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
tf.set_random_seed(444)


p = np.array([[0., 0.], [0., 1.], [0., 2.]], dtype=np.float32)

free_space = np.array([[-2., 0.], [-1., 0.], [-1., 3.], [-2., 3.]], dtype=np.float32)

plt.plot(p[:, 0], p[:, 1])
fs = np.concatenate([free_space, free_space[:1]], axis=0)
plt.plot(fs[:, 0], fs[:, 1])
#plt.show()

A = p[1, 1] - p[0, 1]
B = p[0, 0] - p[1, 0]
C = p[1, 0] * p[0, 1] - p[0, 0] * p[1, 1]
s = (p[0] + p[1]) / 2
print(A, B, C)
A2_B2 = - B / (A + 1e-10)
AB2 = (tf.zeros_like(A), tf.zeros_like(B))
AB2 = tf.where(tf.equal(B, 0.0), (tf.zeros_like(A), tf.ones_like(B)), AB2)
AB2 = tf.where(tf.equal(A, 0.0), (tf.ones_like(A), tf.zeros_like(B)), AB2)
AB2 = tf.where(tf.logical_or(tf.equal(B, 0.0), tf.equal(A, 0.0)), AB2, (A2_B2, tf.ones_like(A)))
C2 = - AB2[0] * s[0] - AB2[1] * s[1]
AB2 = tf.tile(AB2[tf.newaxis], (4, 1))
C2 = tf.tile(C2[tf.newaxis], (4,))
print(AB2)
print(C2)

fs_0 = free_space
fs_1 = tf.roll(free_space, 1, axis=-2)
fs_A = fs_1[:, 1] - fs_0[:, 1]
fs_B = fs_0[:, 0] - fs_1[:, 0]
fs_AB = tf.stack([fs_A, fs_B], -1)
fs_C = fs_1[:, 0] * fs_0[:, 1] - fs_0[:, 0] * fs_1[:, 1]

print(fs_AB)
print(fs_C)

Cs = tf.stack([C2, fs_C], -1)
M = tf.stack([AB2, fs_AB], axis=-2)
print(M)
print(Cs)
pp = tf.linalg.solve(M + tf.eye(2, dtype=tf.float32)[tf.newaxis] * 1e-10, -Cs[..., tf.newaxis])
pp = pp[..., 0]
print("PP")
print(pp)

# CHECK IF BETWEEN ENDS OF SEGMENT
edge = fs_1 - fs_0
pp_in_fs_0 = pp - fs_0

g = tf.reduce_sum(edge * pp_in_fs_0, -1)
t = tf.reduce_sum(edge * edge, -1)
w = g / (t + 1e-8)
w = tf.where(tf.logical_and(w >= 0, w <= 1), tf.ones_like(w), 1e10 * tf.ones_like(w))  # ignore points outside of edge

print(pp)
print(w)
pp = pp * w[:, tf.newaxis]
print(pp)
dist = tf.linalg.norm(pp - s[tf.newaxis], axis=-1)
dist = tf.reduce_min(dist, -1)
print(dist)
#plt.plot(pp[:, 0], pp[:, 1], '*')
#plt.xlim(-5, 5)
#plt.ylim(-5, 5)
#plt.show()

