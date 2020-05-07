import inspect
import os
import sys
import numpy as np
from matplotlib import pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from models.actor import PlanningNetwork
from models.critic import Critic
from models.environment import Environment
from models.planner import _plot
from utils.replay_buffer import ReplayBuffer

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios

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

    train_ds = train_ds \
        .batch(1) \
        .prefetch(1)

    val_ds = val_ds \
        .batch(1) \
        .prefetch(1)

    # 2. Define model
    actor = PlanningNetwork()
    target_actor = PlanningNetwork()
    critic = Critic()
    target_critic = Critic()
    target_actor.update_weights(actor, 1.)
    target_critic.update_weights(critic, 1.)

    env = Environment()

    replay_buffer = ReplayBuffer(10000)

    # 3. Optimization

    actor_optimizer = tf.train.AdamOptimizer(0.1 * args.eta)
    critic_optimizer = tf.train.AdamOptimizer(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, actor, actor_optimizer)

    MAX_STEPS = 7
    EXAMPLES_WITH_NOISE = 1000
    ACTOR_SLEEP = 0
    GAMMA = 0.99
    TAU = 0.001
    # 5. Run everything
    train_step, val_step = 0, 0
    best_accuracy = 0.0
    experiment_handler.log_training()
    for epoch in range(args.num_epochs):
        # 5.1. Training Loop
        experiment_handler.log_training()
        acc = []
        for i, data in _ds('Train', train_ds, train_size, epoch, args.batch_size):
            fs = data[2]
            min_xy = np.min(fs, axis=(0, 1, 2))
            max_xy = np.max(fs, axis=(0, 1, 2))
            xy = (max_xy - min_xy) * np.random.random((2,)) + min_xy
            th = np.pi * 2 * (np.random.random() - 0.5)
            ddy = 1 * np.random.random()
            data = list(data)
            data[0] = np.array([[xy[0], xy[1], th, ddy]], dtype=np.float32)
            state = tuple(data)
            env.set_state(state)
            critic_losses = []
            actor_losses = []
            episode_reward = 0.
            valids = []
            segments = []
            for k in range(MAX_STEPS):
                state = env.get_state()
                action = actor(state)
                if i < EXAMPLES_WITH_NOISE:
                    noise = np.random.normal(np.zeros((1, 4)), np.array([0.2, 0.2, 0.05, 0.1])[np.newaxis]) # TODO
                    action += noise

                new_state, reward, terminal, valid, path = env.step(action)
                valids.append(valid)
                segments.append(path)

                new_state = (np.array(new_state)[np.newaxis], *state[1:])

                replay_buffer.save([state, action, reward, terminal, new_state])

                episode_reward += reward

                if replay_buffer.size() > args.batch_size:
                    s, a, r, t, s2 = replay_buffer.get_batch(args.batch_size)
                    with tf.GradientTape(persistent=True) as tape:
                        target_a = target_actor(s)
                        target_q = target_critic(s2, target_a)

                        y = tf.stop_gradient(r + GAMMA * (1. - t) * target_q)

                        q = critic(s, a)

                        action = actor(s)
                        q_pi = critic(s, action)

                        critic_loss = tf.keras.losses.mean_squared_error(y, q)
                        critic_losses.append(critic_loss)
                        actor_loss = tf.reduce_mean(-q_pi)
                        actor_losses.append(actor_loss)

                    critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
                    critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables),
                                                     global_step=tf.train.get_or_create_global_step())
                    target_critic.update_weights(critic, TAU)
                    if ACTOR_SLEEP < i:
                        actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
                        actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables),
                                                        global_step=tf.train.get_or_create_global_step())

                        target_actor.update_weights(actor, TAU)

                if k == MAX_STEPS - 1:
                    # 5.1.4 Save logs for particular interval
                    with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, i):
                        tfc.summary.scalar('episode/critic_loss', np.mean(critic_losses), step=i)
                        tfc.summary.scalar('episode/actor_loss', np.mean(actor_losses), step=i)
                        tfc.summary.scalar('episode/reward', episode_reward, step=i)
                    acc.append(tf.cast(np.all(valids), tf.float32))


            # 5.1.5 Update meta variables
            # if i % 10 == 0:
            #    _plot(segments, data, i)
            _plot(segments, data, i)
            experiment_handler.flush()
        epoch_accuracy = tf.reduce_mean(acc)

        # 5.1.6 Take statistics over epoch
        with tfc.summary.always_record_summaries():
            tfc.summary.scalar('epoch/good_paths', epoch_accuracy, step=epoch)



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config-file', action=LoadFromFile, type=open)
    parser.add_argument('--scenario-path', type=str)
    parser.add_argument('--working-path', type=str, default='./working_dir')
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--log-interval', type=int, default=1)
    parser.add_argument('--out-name', type=str)
    parser.add_argument('--eta', type=float, default=1e-3)
    args, _ = parser.parse_known_args()
    main(args)
