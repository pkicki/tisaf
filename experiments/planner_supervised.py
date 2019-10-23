import inspect
import os
import sys
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.planner import plan_loss, _plot, PlanningNetworkMP, Poly, PlanningNetwork
from utils.utils import Environment
from dataset.scenarios import Task, load_map, planning_tensor

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
    free_space = load_map(args.scenario_path)
    env = Environment(free_space, 1. / 4.5)
    train_size = 3
    #p = [tf.Variable([[[2.5, 2.0, 1.57], [7.5, 12.0, 0.0]]], dtype=tf.float32),
    #                tf.Variable([[[2.7, 2.0, 1.57], [7.5, 12.0, 0.0]]], dtype=tf.float32),
    #                tf.Variable([[[2.6, 2.15, 1.57], [7.5, 12.0, 0.0]]], dtype=tf.float32)]
    p = planning_tensor(args.scenario_path)

    # 2. Define model
    model = PlanningNetworkMP(7, (args.batch_size, 6))
    model_gt = PlanningNetworkMP(7, (args.batch_size, 6))

    # 3. Optimization

    eta = tfc.eager.Variable(args.eta)
    eta_f = tf.train.exponential_decay(
        args.eta,
        tf.train.get_or_create_global_step(),
        int(float(train_size) / args.batch_size),
        args.train_beta)
    eta.assign(eta_f())
    optimizer = tf.train.AdamOptimizer(eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    eh = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model_gt, optimizer)
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    #eh.restore("./results/test_generalizacji/3_biggernet/checkpoints/last_n-2525")
    #eh.restore("./working_dir/planner_net_3/checkpoints/last_n-400")
    #eh.restore("./working_dir/planner_net_4/checkpoints/last_n-333")
    #eh.restore("./working_dir/planner_net_5/checkpoints/last_n-370")
    #eh.restore("./working_dir/planner_net_6/checkpoints/last_n-303")
    eh.restore("./working_dir/planner_net_10/checkpoints/last_n-7000")

    experiment_handler.restore("./working_dir/planner_net_save/checkpoints/last_n-1000")

    training_set = []
    for x in p:
        x = tf.Variable(x, dtype=tf.float32)[tf.newaxis]
        output, last_ddy = model_gt(x, training=True)
        print(output)
        training_set.append((x, output, last_ddy))

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        # 5.1. Training Loop
        accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        experiment_handler.log_training()
        acc = []
        for data, gt_output, gt_last_ddy in training_set:
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                output, last_ddy = model(data, training=True)
                model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(output, last_ddy, data, env)
                #reg_loss = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                #total_loss = tf.reduce_mean(model_loss)  # + reg_loss
                output_loss = tf.keras.losses.mean_absolute_error(gt_output, output)
                last_ddy_loss = tf.keras.losses.mean_absolute_error(gt_last_ddy, last_ddy)
                total_loss = output_loss + last_ddy_loss

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
                tfc.summary.scalar('metrics/total_loss', total_loss, step=train_step)
                tfc.summary.scalar('metrics/invalid_loss', invalid_loss, step=train_step)
                tfc.summary.scalar('metrics/overshoot_loss', overshoot_loss, step=train_step)
                tfc.summary.scalar('metrics/curvature_loss', curvature_loss, step=train_step)
                tfc.summary.scalar('metrics/balance_loss', non_balanced_loss, step=train_step)
                #tfc.summary.scalar('metrics/reg_loss', reg_loss, step=train_step)
                tfc.summary.scalar('metrics/really_good_paths', s, step=train_step)
                tfc.summary.scalar('metrics/good_paths', t, step=train_step)
                tfc.summary.scalar('training/eta', eta, step=train_step)

            # 5.1.5 Update meta variables
            eta.assign(eta_f())
            train_step += 1
            #if train_step % 20 == 0:
            #    _plot(x_path, y_path, th_path, env, train_step)
            #print(total_loss)
            _plot(x_path, y_path, th_path, env, train_step)
        epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/good_paths', epoch_accuracy, step=epoch)

        #    accuracy.result()

        # 5.2. Validation Loop
        # accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        #experiment_handler.log_validation()
        #acc = []
        #for i, data in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
        #    # 5.2.1 Make inference of the model for validation and calculate losses
        #    #output = model(task, training=False)
        #    #model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(output, task, env)
        #    output, last_ddy = model(data, training=False)
        #    model_loss, invalid_loss, overshoot_loss, curvature_loss, non_balanced_loss, x_path, y_path, th_path = plan_loss(
        #        output, last_ddy, data, env)

        #    t = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
        #    s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))
        #    acc.append(tf.cast(tf.equal(invalid_loss + curvature_loss, 0.0), tf.float32))

        #    # 5.1.2 Calculate statistics
        #    # prediction = tf.argmax(output, -1, output_type=tf.int32)
        #    # accuracy(prediction, labels)
        #    # batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

        #    # 5.2.3 Print logs for particular interval
        #    with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, val_step):
        #        tfc.summary.scalar('metrics/model_loss', model_loss, step=val_step)
        #        tfc.summary.scalar('metrics/invalid_loss', invalid_loss, step=val_step)
        #        tfc.summary.scalar('metrics/overshoot_loss', overshoot_loss, step=val_step)
        #        tfc.summary.scalar('metrics/curvature_loss', curvature_loss, step=val_step)
        #        tfc.summary.scalar('metrics/balance_loss', non_balanced_loss, step=val_step)
        #        tfc.summary.scalar('metrics/really_good_paths', s, step=val_step)
        #        tfc.summary.scalar('metrics/good_paths', t, step=val_step)

        #    # 5.2.4 Update meta variables
        #    val_step += 1

        #epoch_accuracy = tf.reduce_mean(tf.concat(acc, -1))

        ## 5.2.5 Take statistics over epoch
        #with tfc.summary.always_record_summaries():
        #    tfc.summary.scalar('epoch/good_paths', epoch_accuracy, step=epoch)

        ## 5.3 Save last and best
        #if epoch_accuracy > best_accuracy:
        #    experiment_handler.save_best()
        #    best_accuracy = epoch_accuracy
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
    parser.add_argument('--train-beta', type=float, default=0.99)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args, _ = parser.parse_known_args()
    main(args)
