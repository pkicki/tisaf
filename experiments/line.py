import inspect
import os
import sys
import numpy as np

from models.lines import Line

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.lines import plan_loss, _plot, PlanningNetworkMP
from utils.utils import Environment
from dataset.scenarios import Task

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
    train_ds, train_size, free_space = scenarios.planning_dataset(args.scenario_path)
    val_ds, val_size, _ = scenarios.planning_dataset(args.scenario_path)
    #env = Environment(free_space, 1. / 2.57 * np.tan(np.pi * 50 / 180))
    #env = Environment(free_space, 1. / 5.3)
    env = Environment(free_space, 1. / 2.3)

    #train_ds = train_ds \
    #    .batch(args.batch_size) \
    #    .prefetch(args.batch_size)

    val_ds = val_ds \
        .batch(args.batch_size) \
        .prefetch(args.batch_size)

    # 2. Define model
    model = PlanningNetworkMP(8, (args.batch_size, 6))
    #model = PlanningNetwork(3, (args.batch_size, 6))
    #model = Line(3, (args.batch_size, 6))

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
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    #experiment_handler.restore("./results/straight/checkpoints/last_n-50")

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):
        # workaround for tf problems with shuffling
        dataset_epoch = train_ds.shuffle(train_size)
        dataset_epoch = dataset_epoch.batch(args.batch_size).prefetch(args.batch_size)
        #dataset_epoch = train_ds

        # 5.1. Training Loop
        accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        experiment_handler.log_training()
        acc = []
        for i, data in _ds('Train', dataset_epoch, train_size, epoch, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape(persistent=True) as tape:
                output = model(data, training=True)

                #label = tf.ones_like(output)[:, :1, :]
                #label = tf.concat([4.0 * label, label * 0.0], 1)
                #model_loss = tf.keras.losses.mean_absolute_error(label, output)
                #total_loss = model_loss
                #invalid_loss = model_loss

                model_loss, invalid_loss, curv_loss, last_length_loss, some_loss, x_path, y_path, th_path = plan_loss(output, data, env)
                #reg_loss = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                total_loss = tf.reduce_mean(model_loss)  # + reg_loss

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)
            #g1 = tape.gradient(invalid_loss, model.trainable_variables)
            #print(g1)

            #grads = [tf.clip_by_value(g, -1., 1.) for g in grads]
            #grads = [tf.clip_by_norm(g, 1.) for g in grads]
            #print("AFTER:", grads[0])
            #print("LOSS", total_loss)
            #print("GRADS")
            #for k, n in enumerate(model.trainable_variables):
            #    print(i, n.name)
            ##    print(n)
            #    print(grads[k])
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            # 5.1.3 Calculate statistics
            s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
            acc.append(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
            # prediction = tf.argmax(output, -1, output_type=tf.int32)
            # accuracy(prediction, labels)
            # batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

            # 5.1.4 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=train_step)
                tfc.summary.scalar('metrics/invalid_loss', invalid_loss, step=train_step)
                tfc.summary.scalar('metrics/last_length_loss', last_length_loss, step=train_step)
                tfc.summary.scalar('metrics/curvature_loss', curv_loss, step=train_step)
                tfc.summary.scalar('metrics/some_loss', some_loss, step=train_step)
                #tfc.summary.scalar('metrics/reg_loss', reg_loss, step=train_step)
                tfc.summary.scalar('metrics/good_paths', s, step=train_step)
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
        experiment_handler.log_validation()
        acc = []
        #for i, task in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
        #    # 5.2.1 Make inference of the model for validation and calculate losses
        #    output = model(task, training=False)
        #    model_loss, invalid_loss, overshoot_loss, curvature_loss, x_path, y_path, th_path = plan_loss(output, task, env)

        #    s = tf.reduce_mean(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))
        #    acc.append(tf.cast(tf.equal(invalid_loss, 0.0), tf.float32))

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
        #        tfc.summary.scalar('metrics/good_paths', s, step=val_step)

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
        #experiment_handler.save_last()

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
