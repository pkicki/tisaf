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
    experiment_handler.restore("./working_dir/planner_net_4/checkpoints/best-8664")

    # 5. Run everything
    free_space = free_space[tf.newaxis]
    x = -10.0 * tf.ones((1, 1))
    y = -10.0 * tf.ones((1, 1))
    th = 0.3 * tf.ones((1, 1))
    p0 = tf.concat([x, y, th], -1)
    xk = 0. * tf.ones_like(x)
    yk = 0. * tf.ones_like(x)
    thk = 0. * tf.ones_like(x)
    pk = tf.concat([xk, yk, thk], -1)
    data = (p0, pk, free_space)
    with tf.GradientTape(persistent=True) as t:
        output, last_ddy = model(data, None, training=True)
        model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(
            output, last_ddy, data)
    ml = t.gradient(model_loss, output)
    il = t.gradient(invalid_loss, output)
    ol = t.gradient(overshoot_loss, output)
    cl = t.gradient(curvature_loss, output)
    bl = t.gradient(non_balanced_loss, output)
    print("MODEL:", ml)
    print("INVA:", il)
    print("OVER:", ol)
    print("CURV:", cl)
    print("BAL:", bl)


    _plot(x_path, y_path, th_path, data, 0, print=True)

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
