import inspect
import os
import sys
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
from dl_work.utils import ExperimentHandler

tf.enable_eager_execution()
tf.set_random_seed(444)


def main():
    # 1. Get datasets
    ds = scenarios.planning("../../TG_data/train/mix3/")

    # 2. Define model
    model = PlanningNetworkMP(7, (1, 6))

    # 3. Optimization

    optimizer = tf.train.AdamOptimizer(1)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(".", "", 1, model, optimizer)

    #experiment_handler.restore("./paper/tunel_prostopadle/checkpoints/best-6691")
    #experiment_handler.restore("./monster/planner_net_mix/checkpoints/best-6527")
    experiment_handler.restore("./monster/last_mix/checkpoints/best-7274")

    acc = []
    for i in range(len(ds)):
        p0, pk, map = ds[i]
        map = tf.tile(map[tf.newaxis], (len(p0), 1, 1, 1))
        data = (p0, pk, map)
        # 5.2.1 Make inference of the model for validation and calculate losses
        output, last_ddy = model(data, None, training=True)
        model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path, length = plan_loss(
            output, last_ddy, data)

        #_plot(x_path, y_path, th_path, data, i)
        #print(i)
        #print(invalid_loss)
        #print(curvature_loss)
        #print()
        #print()

        acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss + overshoot_loss, 0.0), tf.float32))

    epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))
    print(epoch_accuracy)

main()
