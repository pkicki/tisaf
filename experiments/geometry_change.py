import inspect
import os
import sys
import numpy as np
from experiments.accessible_set_train import _plot_car
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from utils.crucial_points import calculate_car_crucial_points

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import plan_loss, _plot, PlanningNetworkMP

from argparse import ArgumentParser

import tensorflow as tf
from dl_work.utils import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
tf.set_random_seed(444)


def _plot(x_path, y_path, th_path, data, step, print=False):
    _, _, free_space = data
    for k in range(len(x_path[0])):
        plt.figure(num=None, figsize=(9, 2), dpi=300, facecolor='w', edgecolor='k')
        plt.fill([-100., 100., 100., -100., -100.], [-100., -100., 100., 100., -100.], 'brown', zorder=1)
        plt.xlim(-33., 4.5)
        plt.ylim(-0.25, 5.75)
        for i in range(len(x_path)):
            x = x_path[i][k]
            y = y_path[i][k]
            #plt.plot(x, y)
            th = th_path[i][k]
            cp = calculate_car_crucial_points(x, y, th)
            for p in cp:
                plt.plot(p[:, 0], p[:, 1])

        m = free_space[k]
        seq = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (2, 0), (2, 1), (1, 0), (1, 1), (0, 0)]
        x = [m[s][0] for s in seq]
        y = [m[s][1] for s in seq]
        plt.fill(x, y, 'w', zorder=2)

        i = len(x_path) - 1
        x = x_path[i][k]
        y = y_path[i][k]
        th = th_path[i][k]
        cp = calculate_car_crucial_points(x, y, th)[1:]
        _plot_car(cp, -1, 'r')
        i = 0
        x = x_path[i][k]
        y = y_path[i][k]
        th = th_path[i][k]
        cp = calculate_car_crucial_points(x, y, th)[1:]
        _plot_car(cp, 0, 'g')
        if print:
            plt.show()
        else:
            #plt.savefig("last_path" + str(k).zfill(6) + ".pdf")
            plt.savefig("last_path" + str(k).zfill(6) + ".png")
            plt.clf()


def main(args):
    # 1. Get datasets
    #train_ds = scenarios.planning(args.scenario_path)
    val_ds = scenarios.planning(args.scenario_path.replace("train", "val"))

    # 2. Define model
    model = PlanningNetworkMP(7, (args.batch_size, 6))

    # 3. Optimization

    optimizer = tf.train.AdamOptimizer(args.eta)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    experiment_handler.restore("./monster/last_mix/checkpoints/best-7274")

    # 5.2. Validation Loop
    #for i in range(len(val_ds)):
    #q1 = [[-32., -2.], [-19., -2.], [-19., 5.], [-32., 5.]]
    #q2 = [[-10., -2.], [4., -2.], [4., 5.], [-10., 5.]]
    #y = 0.
    #w = 3.0
    #q3 = [[-19., y], [-10., y], [-10., y+w], [-19., y+w]]
    #map = np.array([q1, q2, q3], dtype=np.float32)[np.newaxis]
    n = 20
    r = 0.1
    y = 0.8
    w = 2.7
    map = val_ds[0][2].numpy()[np.newaxis]
    map = np.tile(map, (n + 1, 1, 1, 1))
    rng = np.linspace(y, y + r * n, n + 1)
    map[:, 1, 0, 1] = rng
    map[:, 1, 1, 1] = rng
    map[:, 1, 2, 1] = rng + w
    map[:, 1, 3, 1] = rng + w
    map[:, 0, :2, 1] = 0.
    map[:, 0, 2:, 1] = 5.5
    map[:, 2, :2, 1] = 0.
    map[:, 2, 2:, 1] = 5.5

    map[:, 2, 1, 0] = map[:, 2, 2, 0]
    map[:, 1, 0, 0] = map[:, 2, 1, 0]
    map[:, 1, 3, 0] = map[:, 2, 1, 0]

    map[:, 0, 0, 0] = map[:, 0, 3, 0]
    map[:, 1, 1, 0] = map[:, 0, 0, 0]
    map[:, 1, 2, 0] = map[:, 0, 0, 0]
    p0 = np.array([[-27., 1.5, 0.]], dtype=np.float32)
    p0 = np.tile(p0, (n + 1, 1))
    pk = np.array([[-3., 1.5, 0.]], dtype=np.float32)
    pk = np.tile(pk, (n + 1, 1))
    for i in [1]:
        data = (p0, pk, map)
        # 5.2.1 Make inference of the model for validation and calculate losses
        output, last_ddy = model(data, None, training=True)
        model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(
            output, last_ddy, data)
        print(invalid_loss, curvature_loss)

        _plot(x_path, y_path, th_path, data, 0, False)
        #_plot(x_path, y_path, th_path, data, 0, True)


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
