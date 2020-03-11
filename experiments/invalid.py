import inspect
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import tensorflow as tf

tf.enable_eager_execution()
tf.set_random_seed(444)


bs = 8
p = np.array([[0., 0.], [0., 1.], [0., 2.], [-0.3, 2.5]], dtype=np.float32)
p = tf.tile(p[tf.newaxis, tf.newaxis], (bs, 5, 1, 1))

free_space = np.array([[[-2., 0.], [-1., 0.], [-1., 3.], [-2., 3.]], [[3., 0.], [2., 0.], [2., 3.], [3., 3.]]], dtype=np.float32)
free_space = tf.tile(free_space[tf.newaxis], (bs, 1, 1, 1))
quads_len = free_space.shape[1]

plot_p = p[0]
plt.plot(plot_p[:, 0], plot_p[:, 1])
for i in range(quads_len):
    fs = np.concatenate([free_space[0, i], free_space[0, i, :1]], axis=0)
    plt.plot(fs[:, 0], fs[:, 1])

p_0 = p[:, :, :-1]
p_1 = p[:, :, 1:]
pts_len = tf.shape(p_0)[1]
seq_len = tf.shape(p_0)[2]
p_0_x = p_0[..., 0]
p_0_y = p_0[..., 1]
p_1_x = p_1[..., 0]
p_1_y = p_1[..., 1]
A = p_1_y - p_0_y
B = p_0_x - p_1_x
C = p_1_x * p_0_y - p_0_x * p_1_y
s = (p_0 + p_1) / 2
A2_B2 = - B / (A + 1e-10)
AB2 = (tf.zeros_like(A), tf.zeros_like(B))
AB2 = tf.where((tf.equal(B, 0.0), tf.equal(B, 0.0)), (tf.zeros_like(A), tf.ones_like(B)), AB2)
AB2 = tf.where((tf.equal(A, 0.0), tf.equal(A, 0.0)), (tf.ones_like(A), tf.zeros_like(B)), AB2)
A_or_B = tf.logical_or(tf.equal(B, 0.0), tf.equal(A, 0.0))
AB2 = tf.where((A_or_B, A_or_B), AB2, (A2_B2, tf.ones_like(A)))
AB2 = tf.transpose(AB2, (1, 2, 3, 0))
C2 = - AB2[..., 0] * s[..., 0] - AB2[..., 1] * s[..., 1]

AB2 = tf.tile(AB2[:, tf.newaxis, tf.newaxis], (1, quads_len, 4, 1, 1, 1))
C2 = tf.tile(C2[:, tf.newaxis, tf.newaxis], (1, quads_len, 4, 1, 1))

free_space = tf.tile(free_space[:, :, :, tf.newaxis, tf.newaxis], (1, 1, 1, pts_len, seq_len, 1))
fs_0 = free_space
fs_1 = tf.roll(free_space, 1, axis=2)
fs_0_x = fs_0[..., 0]
fs_0_y = fs_0[..., 1]
fs_1_x = fs_1[..., 0]
fs_1_y = fs_1[..., 1]
fs_A = fs_1_y - fs_0_y
fs_B = fs_0_x - fs_1_x
fs_AB = tf.stack([fs_A, fs_B], -1)
fs_C = fs_1_x * fs_0_y - fs_0_x * fs_1_y

Cs = tf.stack([C2, fs_C], -1)
M = tf.stack([AB2, fs_AB], -2)
pp = tf.linalg.solve(M + tf.eye(2, dtype=tf.float32)[tf.newaxis] * 1e-10, -Cs[..., tf.newaxis])
pp = pp[..., 0]

# CHECK IF BETWEEN ENDS OF SEGMENT
edge = fs_1 - fs_0
pp_in_fs_0 = pp - fs_0

g = tf.reduce_sum(edge * pp_in_fs_0, -1)
t = tf.reduce_sum(edge * edge, -1)
w = g / (t + 1e-8)
w = tf.where(tf.logical_and(w >= 0, w <= 1), tf.ones_like(w), 1e10 * tf.ones_like(w))  # ignore points outside of edge

pp = pp * w[..., tf.newaxis]
dist = tf.linalg.norm(pp - s[:, tf.newaxis, tf.newaxis], axis=-1)
dist = tf.reduce_min(dist, 2) # min from edges
dist = tf.reduce_min(dist, 1) # min from quads
dist = tf.reduce_sum(dist, 1) # for all distinguished points of the vehicle

pp = tf.reshape(pp, (-1, 2))
plt.plot(pp[:, 0], pp[:, 1], '*')
plt.xlim(-5, 5)
plt.ylim(-5, 5)
plt.show()

