import inspect
import os
import sys
from time import time

import numpy as np
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from utils.crucial_points import calculate_car_crucial_points

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner_test import plan_loss, _plot, PlanningNetworkMP

import tensorflow as tf

from utils.execution import ExperimentHandler

tf.enable_eager_execution()
tf.set_random_seed(444)


def main():
    # 1. Get datasets
    ds = scenarios.planning_test("../data/test")

    # 2. Define model
    model = PlanningNetworkMP(7, (1, 6))

    # 3. Optimization

    optimizer = tf.train.AdamOptimizer(1)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(".", "", 1, model, optimizer)

    experiment_handler.restore("./monster/last_mix/checkpoints/best-7274")

    acc = []
    t = []
    t_sl = []
    t_rrt = []
    l = []
    l_rrt = []
    for i in range(len(ds)):#range(1):
        p0, pk, map, rrt, sl = ds[i]
        bs = p0.shape[0]
        for j in range(bs):
            data = (p0[tf.newaxis, j], pk[tf.newaxis, j], map[tf.newaxis])
        #map = tf.tile(map[tf.newaxis], (len(p0), 1, 1, 1))
        #data = (p0, pk, map)
        # 5.2.1 Make inference of the model for validation and calculate losses
            start = time()
            output, last_ddy = model(data, None, training=True)
            end = time()
            rt = end - start
            model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path, length = plan_loss(
                output, last_ddy, data)

            dl = (tf.reduce_sum(length, -1) / sl[j, -1])[0]
            dl_rrt = (1.5 * rrt[j, -1] / sl[j, -1])
            _plot(x_path, y_path, th_path, data, i)
            #print(i, j)
            #print(invalid_loss)
            #print(curvature_loss)
            #print(dl)
            #print(rt)
            #print()

            l.append(dl)
            l_rrt.append(dl_rrt)
            t.append(rt)
            t_sl.append(sl[j, 0])
            t_rrt.append(rrt[j, 0])
            acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss + overshoot_loss, 0.0), tf.float32)[0])

    t = np.array(t)
    t = np.extract(t > 0, t)
    t_sl = np.array(t_sl)
    acc_sl = np.mean((t_sl > 0).astype(np.float32))
    t_sl = np.extract(t_sl > 0, t_sl)
    t_rrt = np.array(t_rrt)
    acc_rrt = np.mean((t_rrt > 0).astype(np.float32))
    t_rrt = np.extract(t_rrt > 0, t_rrt)
    l = np.array(l)
    l = np.extract(l > 0, l)
    l_rrt = np.array(l_rrt)
    l_rrt = np.extract(l_rrt > 0, l_rrt)
    epoch_accuracy = tf.reduce_mean(tf.stack(acc, -1))
    print("ACC:", epoch_accuracy)
    print("T:", np.mean(t[1:]), np.std(t[1:]))
    print("T_RRT:", np.mean(t_rrt[1:]), np.std(t_rrt[1:]))
    print("T_SL:", np.mean(t_sl[1:]), np.std(t_sl[1:]))
    print("L:", np.mean(l), np.std(l))
    print("L_RRT:", np.mean(l_rrt), np.std(l_rrt))
    print("ACC_RRT:", acc_rrt)
    print("ACC_SL:", acc_sl)

main()
