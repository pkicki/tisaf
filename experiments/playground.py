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


class Poly(tf.keras.Model):
    def __init__(self):
        super(Poly, self).__init__()
        t = 5.0
        self.x = tf.Variable(t, trainable=True, name="x1")
        self.y = tf.Variable(0.0, trainable=True, name="y1")
        self.dy = tf.Variable(0.0, trainable=True, name="dy1")
        self.ddy = tf.Variable(0.0, trainable=True, name="ddy1")

        self.x1 = tf.Variable(t, trainable=True, name="x2")
        self.y1 = tf.Variable(0.0, trainable=True, name="y2")
        self.dy1 = tf.Variable(0.0, trainable=True, name="dy2")
        self.ddy1 = tf.Variable(0.0, trainable=True, name="ddy2")

        self.x2 = tf.Variable(t, trainable=True, name="x3")
        self.y2 = tf.Variable(0.0, trainable=True, name="y3")
        self.dy2 = tf.Variable(0.0, trainable=True, name="dy3")
        self.ddy2 = tf.Variable(0.0, trainable=True, name="ddy3")

        self.x3 = tf.Variable(t, trainable=True, name="x4")
        self.y3 = tf.Variable(0.0, trainable=True, name="y4")
        self.dy3 = tf.Variable(0.0, trainable=True, name="dy4")
        self.ddy3 = tf.Variable(0.0, trainable=True, name="ddy4")

        self.ddy4 = tf.Variable([[0.0]], trainable=True, name="ddy5")

    def call(self, task, training=None):
        n = 1
        p = tf.stack([self.x, self.y, self.dy, self.ddy], -1)[tf.newaxis, :, tf.newaxis]
        p = tf.tile(p, [n, 1, 1])
        p1 = tf.stack([self.x1, self.y1, self.dy1, self.ddy1], -1)[tf.newaxis, :, tf.newaxis]
        p1 = tf.tile(p1, [n, 1, 1])
        p2 = tf.stack([self.x2, self.y2, self.dy2, self.ddy2], -1)[tf.newaxis, :, tf.newaxis]
        p2 = tf.tile(p2, [n, 1, 1])
        p3 = tf.stack([self.x3, self.y3, self.dy3, self.ddy3], -1)[tf.newaxis, :, tf.newaxis]
        p3 = tf.tile(p3, [n, 1, 1])
        p = tf.concat([p, p1, p2, p3], -1)
        return p, self.ddy4


def main():
    # 2. Define model
    #data = np.loadtxt("../../TG_data/train/mix3/0.map", delimiter=' ', dtype=np.float32)
    #free_space = np.reshape(data, (1, 3, 4, 2))
    #free_space[0, 0, 0, 0] = -11.
    #free_space[0, 0, 3, 0] = -11.
    #free_space[0, 2, 1:3, 0] = -10.
    #free_space[0, 1, :2, 1] = 2.5
    free_space = plt.imread("map.png")
    p0 = np.array([[3., 2., 0.]], dtype=np.float32)
    pk = np.array([[25., 2., 0.]], dtype=np.float32)
    path = np.array([[3., 2.], [10., 5.], [20., 5.], [25., 2.]], dtype=np.float32)[tf.newaxis]
    data = (p0, pk, free_space, path)
    model = Poly()

    # 3. Optimization
    optimizer = tf.train.AdamOptimizer(1e-2)

    # 5. Run everythin
    for k in range(100):
        with tf.GradientTape(persistent=True) as tape:
            output, last_ddy = model(data, training=True)
            print(output)
            model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(
                output, last_ddy, data)
            total_loss = model_loss
            print(k, "IL", invalid_loss)
            print(k, "CL", curvature_loss)
            print(k, "OL", overshoot_loss)

        # 5.1.2 Take gradients (if necessary apply regularization like clipping),

        # plain gradient
        #grads = tape.gradient(total_loss, model.trainable_variables)
        #for g, v in zip(grads, model.trainable_variables):
        #    print(v.name, g)

        # Gradient clipping
        #t = 1e-2
        grads = tape.gradient(total_loss, model.trainable_variables)
        #grads = [tf.clip_by_value(g, -t, t) for g in grads]

        # BSP
        #t = 1e-1
        g_l = tape.gradient(invalid_loss, model.trainable_variables)
        for g, v in zip(g_l, model.trainable_variables):
            print("INV", v.name, g)
        c_l = tape.gradient(curvature_loss, model.trainable_variables)
        #for g, v in zip(c_l, model.trainable_variables):
        #    print("CUR", v.name, g)
        o_l = tape.gradient(overshoot_loss, model.trainable_variables)
        #for g, v in zip(o_l, model.trainable_variables):
        #    print("OVE", v.name, g)
        b_l = tape.gradient(non_balanced_loss, model.trainable_variables)
        #for g, v in zip(b_l, model.trainable_variables):
        #    print("NB", v.name, g)
        #g_m = [tf.reduce_max(tf.abs(g)) / t for g in g_l]
        #c_m = [tf.reduce_max(tf.abs(g)) / t for g in g_l]
        #o_m = [tf.reduce_max(tf.abs(g)) / t for g in g_l]
        #b_m = [tf.reduce_max(tf.abs(g)) / t for g in g_l]
        #m = [tf.reduce_max(tf.stack([g_m[i], c_m[i], o_m[i], b_m[i], 1.])) for i in range(len(g_m))]
        #for i in range(len(g_l)):
        #    #print("I:", g_l[i])
        #    #print("C:", c_l[i])
        #    #print("O:", o_l[i])
        #    if o_l[i] is None:
        #        o_l[i] = tf.Variable([[0.0]], dtype=tf.float32)
        #    #print("B:", b_l[i])
        #grads = [(g_l[i] + c_l[i] + o_l[i] + b_l[i]) / m[i] for i in range(len(g_l))]

        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())

        _plot(x_path, y_path, th_path, data, k)


if __name__ == '__main__':
    main()
