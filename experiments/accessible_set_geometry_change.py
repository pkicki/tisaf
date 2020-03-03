import inspect
import os
import sys
from random import shuffle
from time import time

import numpy as np
from matplotlib import pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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
#tf.set_random_seed(444)

def _plot_car(cp, idx, c='m', alpha=0.5):
    x = [p[idx, 0].numpy() for p in cp]
    y = [p[idx, 1].numpy() for p in cp]
    tmp = x[2]
    x[2] = x[3]
    x[3] = tmp
    tmp = y[2]
    y[2] = y[3]
    y[3] = tmp
    plt.fill(x + [x[0]], y + [y[0]], c, alpha=alpha, zorder=5)


def _plot(x_path, y_path, th_path):
    k = 0
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
            plt.plot(p[:, 0], p[:, 1], zorder=6)

    i = len(x_path) - 1
    x = x_path[i][k]
    y = y_path[i][k]
    th = th_path[i][k]
    cp = calculate_car_crucial_points(x, y, th)[1:]
    _plot_car(cp, -1, 'r', 0.5)
    i = 0
    x = x_path[i][k]
    y = y_path[i][k]
    th = th_path[i][k]
    cp = calculate_car_crucial_points(x, y, th)[1:]
    _plot_car(cp, 0, 'g', 0.5)

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

    experiment_handler.restore("./paper/tunel_prostopadle/checkpoints/best-6691")

    i = -2

    a = 60
    x = np.linspace(-32.0, -20., a)
    b = 40
    y = np.linspace(5.0, 0.5, b)
    c = 91
    d = 4
    th = np.linspace(-np.pi / d, np.pi / d,  c)
    X, Y, TH = np.meshgrid(x, y, th)
    x, y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    TH = TH.flatten()
    p0 = np.stack([X, Y, TH], 1).astype(np.float32)
    n = a * b * c

    r = 0.1
    w = 2.7
    map = val_ds[0][2].numpy()[np.newaxis]
    map = np.tile(map, (n, 1, 1, 1))
    rng = 0.8 + 2 * r
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

    for w in [17]:#, 17]:#range(30):
        pk = np.array([[-3., 1.5, 0.]], dtype=np.float32)
        pk = np.tile(pk, (n, 1))

        data = (p0, pk, map)
        # 5.2.1 Make inference of the model for validation and calculate losses
        pp0 = np.array([[-30., 1.5, 0.]], dtype=np.float32)
        ppk = np.array([[0., 1.5, 0.]], dtype=np.float32)
        dummy_data = (pp0, ppk, map[:1])
        output, last_ddy = model(dummy_data, None, training=True)
        _, _, _, _, _, px_path, py_path, pth_path = plan_loss(
            output, last_ddy, dummy_data)
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
        plt.figure(num=None, figsize=(9, 2), dpi=300, facecolor='w', edgecolor='k')
        plt.fill([-100., 100., 100., -100., -100.], [-100., -100., 100., 100., -100.], 'brown', zorder=1)
        plt.xlim(-33., 4.5)
        plt.ylim(-0.25, 5.75)

        m = map[0]
        #seq = [(1, 3), (1, 0), (0, 3), (0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (2, 0), (1, 3)]
        seq = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 2), (2, 3), (2, 0), (2, 1), (1, 0), (1, 1), (0, 0)]
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
        plt.scatter(tf.reshape(x, [-1])[::-1], tf.reshape(y, [-1])[::-1], c=color[::-1], s=1.5*np.ones_like(color), zorder=4, cmap='hot_r')
        plt.colorbar(orientation="horizontal")
        print(w, "PK:", pk[0, 0], pk[0, 1])
        _plot(px_path, py_path, pth_path)
        #plt.show()
        #plt.savefig("xD1.pdf")
        plt.savefig("2.pdf")
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
    parser.add_argument('--train-beta', type=float, default=0.99)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args, _ = parser.parse_known_args()
    main(args)
