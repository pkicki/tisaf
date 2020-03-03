import inspect
import os
import sys
from random import shuffle
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
from models.planner import plan_loss, _plot, PlanningNetworkMP

from argparse import ArgumentParser

import tensorflow as tf

from utils.execution import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
#tf.set_random_seed(444)

def _plot_car(cp, idx, c='m'):
    x = [p[idx, 0].numpy() for p in cp]
    y = [p[idx, 1].numpy() for p in cp]
    tmp = x[2]
    x[2] = x[3]
    x[3] = tmp
    tmp = y[2]
    y[2] = y[3]
    y[3] = tmp
    plt.fill(x + [x[0]], y + [y[0]], c, alpha=0.5, zorder=3)


def _plot(x_path, y_path, th_path, gidx):
    shuffle(gidx)
    gidx = list(gidx[:1, 0])
    cl = ['c', 'g', 'b', 'm', 'k']
    for k in gidx:
        for i in range(0, len(x_path)):
            x = x_path[i][k]
            #print("X:", x)
            y = y_path[i][k]
            #print("Y:", y)
            #plt.plot(x, y)
            th = th_path[i][k]
            #print("TH:", th)
            cp = calculate_car_crucial_points(x, y, th)
            for j, p in enumerate(cp):
                plt.plot(p[:, 0], p[:, 1], color=cl[j], zorder=4)
        i = len(x_path) - 1
        x = x_path[i][k]
        y = y_path[i][k]
        th = th_path[i][k]
        cp = calculate_car_crucial_points(x, y, th)[1:]
        #xy = [[p[-1, 0].numpy(), p[-1, 1].numpy()] for p in cp]
        _plot_car(cp, -1)


def main(args):
    # 1. Get datasets
    train_ds = scenarios.planning(args.scenario_path)
    #val_ds = scenarios.planning(args.scenario_path.replace("train", "val"))

    # 2. Define model
    model = PlanningNetworkMP(7, (args.batch_size, 6))

    # 3. Optimization

    optimizer = tf.train.AdamOptimizer(args.eta)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    experiment_handler.restore("./monster/last_mix/checkpoints/best-7274")

    i = -2

    a = 60
    x = np.linspace(-12.5, -3., a)
    b = 30
    y = np.linspace(-21.0, -16., b)
    c = 76
    #d = 4
    #th = np.linspace(-np.pi / d, np.pi / d,  c)
    th = np.linspace(- 10 * np.pi / 180, 65 * np.pi / 180,  c)
    X, Y, TH = np.meshgrid(x, y, th)
    x, y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    TH = TH.flatten()
    p0 = np.stack([X, Y, TH], 1).astype(np.float32)
    n = a * b * c

    map = train_ds[i][2].numpy()[np.newaxis]
    map = np.tile(map, (n, 1, 1, 1))

    for w in [17]:#, 17]:#range(30):
        pk = train_ds[i][1][w].numpy()[np.newaxis]
        pk = np.tile(pk, (n, 1))

        p0a = train_ds[i][0][w].numpy()

        data = (p0, pk, map)
        # 5.2.1 Make inference of the model for validation and calculate losses
        #dummy_data = (p0[:1], pk[:1], map[:1])
        #output, last_ddy = model(dummy_data, None, training=True)
        start = time()
        output, last_ddy = model(data, None, training=True)
        end = time()
        print("TIME:", end - start)
        model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(
            output, last_ddy, data)
        #print(invalid_loss, curvature_loss)

        l = invalid_loss + curvature_loss + overshoot_loss
        gidx = l.numpy()
        gidx = np.argwhere(gidx == 0)
        l = tf.reshape(l, (-1, c))
        color = tf.reduce_sum(tf.cast(tf.equal(l, 0.0), tf.float32), -1)

        #for i in range(map.shape[1]):
        #    for j in range(4):
        #        fs = map
        #        plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]], zorder=1, color='orange')
        #c = 'brown'
        plt.fill([-100., 100., 100., -100., -100.], [-100., -100., 100., 100., -100.], 'brown', zorder=1)
        plt.xlim(-13.25, 8.25)
        plt.ylim(-22., 3.)

        m = map[0]
        seq = [(1, 3), (1, 0), (0, 3), (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (2, 0), (1, 3)]
        plt.fill([m[s][0] for s in seq], [m[s][1] for s in seq], 'w', zorder=2)

        #plt.plot([map[0, 0, -1, 0], map[0, 0, 0, 0]], [map[0, 0, -1, 1], map[0, 0, 0, 1]], zorder=2, color=c)
        #plt.plot([map[0, 0, 0, 0], map[0, 0, 1, 0]], [map[0, 0, 0, 1], map[0, 0, 1, 1]], zorder=2, color=c)
        #plt.plot([map[0, 0, 1, 0], map[0, 0, 2, 0]], [map[0, 0, 1, 1], map[0, 0, 2, 1]], zorder=2, color=c)
        #plt.plot([map[0, 1, -1, 0], map[0, 1, 0, 0]], [map[0, 1, -1, 1], map[0, 1, 0, 1]], zorder=2, color=c)
        #plt.plot([map[0, 1, 0, 0], map[0, 0, -1, 0]], [map[0, 1, 0, 1], map[0, 0, -1, 1]], zorder=2, color=c)
        #plt.plot([map[0, 1, 0, 0], map[0, 1, 1, 0]], [map[0, 0, 2, 1], map[0, 1, 1, 1]], zorder=2, color=c)
        #plt.plot([map[0, 1, 1, 0], map[0, 1, 2, 0]], [map[0, 1, 1, 1], map[0, 1, 2, 1]], zorder=2, color=c)
        #plt.plot([map[0, 1, 2, 0], map[0, 1, 3, 0]], [map[0, 2, 0, 1], map[0, 1, 3, 1]], zorder=2, color=c)
        #plt.plot([map[0, 2, -1, 0], map[0, 2, 0, 0]], [map[0, 2, -1, 1], map[0, 2, 0, 1]], zorder=2, color=c)
        #plt.plot([map[0, 2, 1, 0], map[0, 2, 2, 0]], [map[0, 2, 1, 1], map[0, 2, 2, 1]], zorder=2, color=c)
        #plt.plot([map[0, 2, 2, 0], map[0, 2, 3, 0]], [map[0, 2, 2, 1], map[0, 2, 3, 1]], zorder=2, color=c)
        plt.scatter(tf.reshape(x, [-1])[::-1], tf.reshape(y, [-1])[::-1], c=color[::-1], s=1.5*np.ones_like(color), zorder=3, cmap='hot_r')
        plt.colorbar()
        plt.arrow(pk[0, 0], pk[0, 1], np.cos(pk[0, 2]), np.sin(pk[0, 2]), width=0.1, zorder=10, color='r')
        plt.arrow(p0a[0], p0a[1], np.cos(p0a[2]), np.sin(p0a[2]), width=0.2, zorder=11, color='b')
        print(w, "P0:", p0a[0], p0a[1])
        print(w, "PK:", pk[0, 0], pk[0, 1])
        _plot(x_path, y_path, th_path, gidx)
        plt.show()
        #plt.savefig(str(w)+".png")
        #plt.clf()


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
    args, _ = parser.parse_known_args()
    main(args)
