import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from models.planner import _calculate_global_xyth_and_curvature
from utils.poly5 import params
from utils.utils import _calculate_length
import matplotlib.pyplot as plt

tf.enable_eager_execution()

max_curvature = 1 / 4.2


def process_segment(plan, xL, yL, thL, last_ddy):
    x = plan[:, 0]
    tan_th = plan[:, 2]

    # calculate params
    zeros = tf.zeros_like(last_ddy)
    p = params([zeros, zeros, zeros, last_ddy], tf.unstack(plan, axis=1))

    # calculate xy coords of segment
    x_glob, y_glob, th_glob, curvature = _calculate_global_xyth_and_curvature(p, x, xL, yL, thL)

    # calcualte length of segment
    length, segments = _calculate_length(x_glob, y_glob)

    # calculate violations
    curvature_violation = tf.reduce_sum(tf.nn.relu(tf.abs(curvature[:, 1:]) - max_curvature) * segments, -1)
    # curvature_violation = tf.reduce_sum(tf.nn.relu(tf.abs(curvature) - env.max_curvature), -1)
    # curvature_violation = tf.reduce_sum(tf.abs(curvature), -1)
    # curvature_violation = tf.reduce_sum(tf.square(curvature), -1)

    return x_glob, y_glob, th_glob, curvature_violation, length, x_glob[:, -1], y_glob[:, -1], th_glob[:, -1]


xL = np.array([0.0], dtype=np.float32)
yL = np.array([0.0], dtype=np.float32)
thL = np.array([0.0], dtype=np.float32)
last_ddy = np.array([0.0], dtype=np.float32)

optimizer = tf.keras.optimizers.Adam(1e-2)
#optimizer = tf.keras.optimizers.SGD(1e-2)

s = tf.Variable([1.0], trainable=True)

t = []

#for i in range(100):
#    with tf.GradientTape(persistent=True) as tape:
#        print(s)
#        plan = np.array([3.0, 0.8, 0.5], dtype=np.float32)
#        plan = tf.concat([plan, s], 0)[tf.newaxis]
#
#        x_glob, y_glob, th_glob, curvature_violation, length, xL, yL, thL = \
#            process_segment(plan, xL, yL, thL, last_ddy)
#        print(curvature_violation)
#        t.append(curvature_violation)
#
#    grads = tape.gradient(curvature_violation, [s])
#    optimizer.apply_gradients(zip(grads, [s]))

for i in range(-30, 30):
    s = float(i) / 30
    print(s)
    plan = np.array([3.5, 0.7, 0.4, s], dtype=np.float32)[tf.newaxis]

    x_glob, y_glob, th_glob, curvature_violation, length, xL, yL, thL = \
        process_segment(plan, xL, yL, thL, last_ddy)
    print(curvature_violation)
    t.append(curvature_violation)

plt.plot(t)
plt.show()
#plt.plot(x_glob[0], y_glob[0])
#plt.show()
