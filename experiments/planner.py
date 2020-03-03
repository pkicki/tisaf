import inspect
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

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
    train_ds, train_size = scenarios.planning_dataset(args.scenario_path)
    val_ds, val_size = scenarios.planning_dataset(args.scenario_path.replace("train", "val"))

    val_ds = val_ds \
        .batch(args.batch_size) \
        .prefetch(args.batch_size)

    # 2. Define model
    model = PlanningNetworkMP(7, (args.batch_size, 6))

    # 3. Optimization

    eta = tfc.eager.Variable(args.eta)
    eta_f = tf.train.exponential_decay(
        args.eta,
        tf.train.get_or_create_global_step(),
        int(float(train_size) / args.batch_size),
        args.train_beta)
    eta.assign(eta_f())
    optimizer = tf.train.AdamOptimizer(eta)
<<<<<<< HEAD
=======
    l2_reg = tf.keras.regularizers.l2(1e-5)
>>>>>>> 563e74c... IROS models

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        # workaround for tf problems with shuffling
        dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)

        # 5.1. Training Loop
        experiment_handler.log_training()
        acc = []
        for i, data in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                output, last_ddy = model(data, None, training=True)
                model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(output, last_ddy, data)
                total_loss = model_loss

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)

            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            # 5.1.3 Calculate statistics
            t = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
            s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))
            acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))

            # 5.1.4 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=train_step)
                tfc.summary.scalar('metrics/invalid_loss', invalid_loss, step=train_step)
                tfc.summary.scalar('metrics/overshoot_loss', overshoot_loss, step=train_step)
                tfc.summary.scalar('metrics/curvature_loss', curvature_loss, step=train_step)
                tfc.summary.scalar('metrics/balance_loss', non_balanced_loss, step=train_step)
                tfc.summary.scalar('metrics/really_good_paths', s, step=train_step)
                tfc.summary.scalar('metrics/good_paths', t, step=train_step)
                tfc.summary.scalar('training/eta', eta, step=train_step)

            # 5.1.5 Update meta variables
            eta.assign(eta_f())
            train_step += 1
            if train_step % 20 == 0:
                _plot(x_path, y_path, th_path, data, train_step)
        epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/good_paths', epoch_accuracy, step=epoch)

<<<<<<< HEAD

=======
>>>>>>> 563e74c... IROS models
        # 5.2. Validation Loop
        experiment_handler.log_validation()
        acc = []
        for i, data in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
            # 5.2.1 Make inference of the model for validation and calculate losses
            output, last_ddy = model(data, None, training=True)
            model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(
                output, last_ddy, data)

            t = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
            s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))
            acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))

            # 5.2.3 Print logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, val_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=val_step)
                tfc.summary.scalar('metrics/invalid_loss', invalid_loss, step=val_step)
                tfc.summary.scalar('metrics/overshoot_loss', overshoot_loss, step=val_step)
                tfc.summary.scalar('metrics/curvature_loss', curvature_loss, step=val_step)
                tfc.summary.scalar('metrics/balance_loss', non_balanced_loss, step=val_step)
                tfc.summary.scalar('metrics/really_good_paths', s, step=val_step)
                tfc.summary.scalar('metrics/good_paths', t, step=val_step)

            # 5.2.4 Update meta variables
            val_step += 1

        epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        # 5.2.5 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/good_paths', epoch_accuracy, step=epoch)

        # 5.3 Save last and best
        if epoch_accuracy > best_accuracy:
            experiment_handler.save_best()
            best_accuracy = epoch_accuracy
        experiment_handler.save_last()

        experiment_handler.flush()


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
