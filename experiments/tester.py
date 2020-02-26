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
from models.planner import plan_loss, _plot, PlanningNetworkMP
from utils.utils import Environment
from dataset.scenarios import Task
from models.maps import MapAE

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from dl_work.utils import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
tf.set_random_seed(444)


def _plot(x_path, y_path, th_path, data, step, print=False):
    _, _, free_space = data
    for i in range(len(x_path)):
        for k in [2]:#range(len(x_path[i])):
            x = x_path[i][k]
            y = y_path[i][k]
            #plt.plot(x, y)
            th = th_path[i][k]
            cp = calculate_car_crucial_points(x, y, th)
            for p in cp:
                plt.plot(p[:, 0], p[:, 1])

    for i in range(free_space.shape[1]):
        for j in range(4):
            fs = free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    # plt.xlim(-25.0, 25.0)
    # plt.ylim(0.0, 50.0)
    # plt.xlim(-15.0, 20.0)
    # plt.ylim(0.0, 35.0)
    # plt.xlim(-35.0, 5.0)
    # plt.ylim(-2.0, 6.0)
    if print:
        plt.show()
    else:
        plt.savefig("last_path" + str(step).zfill(6) + ".png")
        plt.clf()


def main(args):
    # 1. Get datasets
    train_ds = scenarios.planning(args.scenario_path)
    val_ds = scenarios.planning(args.scenario_path.replace("train", "val"))

    # 2. Define model
    model = PlanningNetworkMP(7, (args.batch_size, 6))

    # 3. Optimization

    optimizer = tf.train.AdamOptimizer(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    experiment_handler.restore("./paper/tunel_prostopadle/checkpoints/best-6691")

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0

    # 5.1. Training Loop
    #acc = []
    #for i in range(len(train_ds)):
    #    p0, pk, map = train_ds[i]
    #    bs = p0.shape[0]
    #    map = tf.tile(map[tf.newaxis], (bs, 1, 1, 1))
    #    data = (p0, pk, map)
    #    # 5.1.1. Make inference of the model, calculate losses and record gradients
    #
    #    output, last_ddy = model(data, None, training=True)
    #    model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(output, last_ddy, data)


    #    # 5.1.3 Calculate statistics
    #    acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))

    #    # 5.1.5 Update meta variables
    #    train_step += 1
    #epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))
    #print(epoch_accuracy)

    #    accuracy.result()

    # 5.2. Validation Loop
    acc = []
    #for i in range(len(val_ds)):
    for i in [1]:
        p0, pk, map = val_ds[i]
        bs = p0.shape[0]
        map = tf.tile(map[tf.newaxis], (bs, 1, 1, 1))
        data = (p0, pk, map)
        # 5.2.1 Make inference of the model for validation and calculate losses
        output, last_ddy = model(data, None, training=True)
        model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(
            output, last_ddy, data)

        for i in range(bs):
            print(i, invalid_loss[i])

        _plot(x_path, y_path, th_path, data, 0, True)
        #_plot(x_path, y_path, th_path, data, 0, True)

        acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))

    epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))
    print(epoch_accuracy)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--scenario-path', type=str)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--train-beta', type=float, default=0.99)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args, _ = parser.parse_known_args()
    main(args)
