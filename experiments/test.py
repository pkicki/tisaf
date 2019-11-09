import inspect
import os
import sys
from math import pi
from time import time

import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from models.planner import plan_loss, _plot, PlanningNetworkMP

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from dl_work.utils import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(ds):
    for i, data in enumerate(ds):
        yield (i,) + (data,)


def main():
    # 2. Define model
    model = PlanningNetworkMP(7, (1, 7))
    # 3. Optimization
    optimizer = tf.train.AdamOptimizer(1)
    # 4. Restore, Log & Save

    ckpt = tf.train.Checkpoint(optimizer=optimizer,
                               model=model,
                               optimizer_step=tf.train.get_or_create_global_step())
    path = "./results/tunel_wytrenowany/best-3460"
    ckpt.restore(path)

    N = 20
    #N = 0
    for i in range(N+1):

        q0 = np.array([5.3644, 1.2775, 1.6512, -0.0910], dtype=np.float32)[np.newaxis]
        # start position x
        q0[:, 0] += (i - N/2) * 0.2
        # start orientation
        #q0[:, 2] += (i - N/2) * 2 * pi / 180
        pk = np.array([4.9934, 23.9864, 1.6065], dtype=np.float32)[np.newaxis]
        # end position x
        #pk[:, 0] += (i - N/2) * 0.2
        # end orientation
        #pk[:, 2] += (i - N/2) * 2 * pi / 180
        quad1 = np.array([[0.00, 0.00], [10.90, 0.00], [10.90, 10.72], [0.00, 10.72]], dtype=np.float32)
        quad2 = np.array([[2.84, 10.72], [6.74, 10.72], [6.74, 18.08], [2.84, 18.08]], dtype=np.float32)
        # tunel position
        #W = quad2[1, 0] - quad2[0, 0]
        #L = (10.90 - W) * i / N
        #R = L + W
        #quad2[0, 0] = L
        #quad2[1, 0] = R
        #quad2[2, 0] = R
        #quad2[3, 0] = L
        quad3 = np.array([[0.00, 18.08], [10.90, 18.08], [10.90, 28.80], [0.00, 28.80]], dtype=np.float32)
        # tunel length
        #T = quad2[2, 1] + (i - N/2) * 0.3
        #quad2[2, 1] = T
        #quad2[3, 1] = T
        #quad3[0, 1] = T
        #quad3[1, 1] = T

        #free_space = np.stack([quad1, quad2, quad3], 0)[np.newaxis]
        free_space = np.stack([quad1, quad2, quad3], 0)[np.newaxis]
        img = None
        data = (q0, pk, free_space, img)

        start = time()
        output, last_ddy = model(data, None, training=True)
        #model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(output,
                                                                                                                     #last_ddy,
                                                                                                                     #data)
        print(time() - start)
        #print(model_loss)
        #_plot(x_path, y_path, th_path, data, i)

if __name__ == '__main__':
    main()
