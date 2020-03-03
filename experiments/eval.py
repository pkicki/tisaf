import inspect
import os
import sys
from math import pi

import numpy as np
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import plan_loss, _plot, PlanningNetworkMP
from utils.utils import Environment
from dataset.scenarios import Task, load_map

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

from dl_work.utils import ExperimentHandler, LoadFromFile

tf.enable_eager_execution()
tf.set_random_seed(444)

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i, data)
            pbar.update(batch_size)


def main(args):
    # 1. Get datasets
    free_space = load_map(args.scenario_path + "/map2.scn")
    env = Environment(free_space, 1. / 4.5)

    # 2. Define model
    model = PlanningNetworkMP(4, (args.batch_size, 6))

    # 3. Optimization

    optimizer = tf.train.AdamOptimizer(1.)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    #experiment_handler.restore("./results/test_generalizacji/1/checkpoints/best-1480")
    #experiment_handler.restore("./results/test_generalizacji/2/checkpoints/last_n-2017")
    #experiment_handler.restore("./results/test_generalizacji/3/checkpoints/last_n-2643")
    #experiment_handler.restore("./results/test_generalizacji/3_biggernet/checkpoints/last_n-2525")
    #experiment_handler.restore("./results/test_generalizacji/3_supervised/checkpoints/last_n-823")
    #experiment_handler.restore("./results/test_generalizacji/3_supervised_slower/checkpoints/last_n-331")
    #experiment_handler.restore("./working_dir/planner_net_/checkpoints/last_n-700")
    experiment_handler.restore("./working_dir/planner_net_4/checkpoints/best-8664")


    # 5. Run everything
    xn = 30
    yn = 30
    thn = 1
    free_space = tf.tile(free_space[tf.newaxis], (xn*yn*thn, 1, 1, 1))
    #x = tf.linspace(2.5, 2.7, xn)[:, tf.newaxis, tf.newaxis]
    x = tf.linspace(-16., -5., xn)[:, tf.newaxis, tf.newaxis]
    #y = tf.linspace(2.0, 2.15, yn)[tf.newaxis, :, tf.newaxis]
    y = tf.linspace(-17., -10., yn)[tf.newaxis, :, tf.newaxis]
    #th = tf.linspace(0.0, 0.0, thn)[tf.newaxis, tf.newaxis, :] + pi / 2
    th = tf.linspace(0.0, 0.0, thn)[tf.newaxis, tf.newaxis, :] #+ 1.57#pi / 2
    x = tf.tile(x, [1, yn, thn])
    y = tf.tile(y, [xn, 1, thn])
    th = tf.tile(th, [xn, yn, 1])
    p0 = tf.stack([x, y, th], -1)
    p0 = tf.reshape(p0, (xn*yn*thn, 3))
    xk = 0. * tf.ones_like(x)
    yk = 0. * tf.ones_like(x)
    thk = 0. * tf.ones_like(x)
    pk = tf.stack([xk, yk, thk], -1)
    pk = tf.reshape(pk, (xn*yn*thn, 3))
    data = (p0, pk, free_space)
    output, last_ddy = model(data, None, training=True)
    model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(output, last_ddy, data)
    print(invalid_loss + curvature_loss + overshoot_loss)
    color = tf.equal(invalid_loss + curvature_loss + overshoot_loss, 0.0)
    print(x)
    print(y)
    plt.scatter(tf.reshape(x, [-1]), tf.reshape(y, [-1]), c=color)
    for i in range(free_space.shape[1]):
        for j in range(4):
            fs = free_space
            plt.plot([fs[0, i, j - 1, 0], fs[0, i, j, 0]], [fs[0, i, j - 1, 1], fs[0, i, j, 1]])
    plt.show()


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
