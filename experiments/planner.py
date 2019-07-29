import inspect
import os
import sys

# add parent (root) to pythonpath
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from argparse import ArgumentParser

import tensorflow as tf
import tensorflow.contrib as tfc
from tqdm import tqdm

import dl_work as dw
from dl_work.utils import ExperimentHandler
from models import GaussianInferenceNetwork, gauss_loss

tf.enable_eager_execution()

_tqdm = lambda t, s, i: tqdm(
    ncols=80,
    total=s,
    bar_format='%s epoch %d | {l_bar}{bar} | Remaining: {remaining}' % (t, i))


def _ds(title, ds, ds_size, i, batch_size):
    with _tqdm(title, ds_size, i) as pbar:
        for i, data in enumerate(ds):
            yield (i,) + data
            pbar.update(batch_size)


def map_ds(rgb, depth, s1, s2, s3, s4, s5, label):
    data = tf.concat([rgb, depth, s1, s2, s3, s4, s5], -1)
    return data, label


def main(args):
    # 1. Get datasets
    train_ds, train_size = dw.dataset.put_multi(args.dataset_path, args.height, args.width, args.augment, True)
    val_ds, val_size = dw.dataset.put_multi(args.dataset_path, args.height, args.width, False, False)

    train_ds = train_ds \
        .map(map_ds, 8) \
        .batch(args.batch_size) \
        .prefetch(args.batch_size)

    val_ds = val_ds \
        .map(map_ds, 8) \
        .batch(args.batch_size) \
        .prefetch(args.batch_size)

    # 2. Define model
    spec = GaussianInferenceNetwork.MoG_spec(5, [3, 1, 1, 1, 1, 1, 1], 7)
    model = GaussianInferenceNetwork(**spec, train_size=train_size, batch_size=args.batch_size)

    # 3. Optimization
    eta = tfc.eager.Variable(args.eta)
    eta_f = tf.train.exponential_decay(
        args.eta,
        tf.train.get_or_create_global_step(),
        int(float(train_size) / args.batch_size),
        args.train_beta)
    eta.assign(eta_f())
    optimizer = tf.train.AdamOptimizer(eta)
    l2_reg = tf.keras.regularizers.l2(1e-4)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, model, optimizer)

    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
    for epoch in range(args.num_epochs):

        # 5.1. Training Loop
        accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        experiment_handler.log_training()
        for i, data, labels in _ds('Train', train_ds, train_size, epoch, args.batch_size):
            # 5.1.1. Make inference of the model, calculate losses and record gradients
            with tf.GradientTape() as tape:
                output = model(data, training=True)
                model_loss = gauss_loss(labels, output)
                reg_loss = tfc.layers.apply_regularization(l2_reg, model.trainable_variables)
                total_loss = model_loss + reg_loss

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                      global_step=tf.train.get_or_create_global_step())

            # 5.1.3 Calculate statistics
            prediction = tf.argmax(output, -1, output_type=tf.int32)
            accuracy(prediction, labels)
            batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

            # 5.1.4 Save logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=train_step)
                tfc.summary.scalar('metrics/reg_loss', reg_loss, step=train_step)
                tfc.summary.scalar('metrics/batch_accuracy', batch_accuracy, step=train_step)
                tfc.summary.histogram('metrics/class_embeddings', model.mod_comb.class_embeddings, step=val_step)

            # 5.1.5 Update meta variables
            eta.assign(eta_f())
            train_step += 1

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            accuracy.result()

        # 5.2. Validation Loop
        accuracy = tfc.eager.metrics.Accuracy('metrics/accuracy')
        experiment_handler.log_validation()
        for i, data, labels in _ds('Validation', val_ds, val_size, epoch, args.batch_size):
            # 5.2.1 Make inference of the model for validation and calculate losses
            output = model(data, training=False)
            model_loss = gauss_loss(labels, output)

            # 5.1.2 Calculate statistics
            prediction = tf.argmax(output, -1, output_type=tf.int32)
            accuracy(prediction, labels)
            batch_accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, labels), tf.float32))

            # 5.2.3 Print logs for particular interval
            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, val_step):
                tfc.summary.scalar('metrics/model_loss', model_loss, step=val_step)
                tfc.summary.scalar('metrics/batch_accuracy', batch_accuracy, step=val_step)

            # 5.2.4 Update meta variables
            val_step += 1

        # 5.2.5 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            epoch_accuracy = accuracy.result()

        # 5.3 Save last and best
        if epoch_accuracy > best_accuracy:
            experiment_handler.save_best()
            best_accuracy = epoch_accuracy
        experiment_handler.save_last()

        experiment_handler.flush()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset-path', type=str, required=True)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int, required=True)
    parser.add_argument('--batch-size', type=int, required=True)
    parser.add_argument('--log-interval', type=int, default=5)
    parser.add_argument('--out-name', type=str, required=True)
    parser.add_argument('--eta', type=float, default=5e-4)
    parser.add_argument('--train-beta', type=float, default=0.99)
    parser.add_argument('--augment', action='store_true', default=False)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args, _ = parser.parse_known_args()
    main(args)
