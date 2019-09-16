import inspect
import os
import random
import sys
from cmath import pi
from copy import copy

import numpy as np

from models.critic import CriticNetwork
from utils.replay_buffer import ReplayBuffer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# add parent (root) to pythonpath
from dataset import scenarios
from models.actor import plan_loss, _plot, PlanningNetworkMP, calculate_next_point
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
    env = Environment(free_space, 1. / 5.3)

    replay_buffer = ReplayBuffer(10000)

    # 2. Define model
    actor = PlanningNetworkMP((args.batch_size, 7))
    target_actor = PlanningNetworkMP((args.batch_size, 7))
    for i in range(len(actor.trainable_variables)):
        tf.assign(target_actor.trainable_variables[i], actor.trainable_variables[i])

    critic = CriticNetwork()
    target_critic = CriticNetwork()
    for i in range(len(critic.trainable_variables)):
        tf.assign(target_critic.trainable_variables[i], critic.trainable_variables[i])

    # 3. Optimization
    actor_optimizer = tf.train.AdamOptimizer(args.eta)
    critic_optimizer = tf.train.AdamOptimizer(args.eta)
    l2_reg = tf.keras.regularizers.l2(1e-5)

    # 4. Restore, Log & Save
    experiment_handler = ExperimentHandler(args.working_path, args.out_name, args.log_interval, actor, actor_optimizer)

    num_episodes = int(1e8)
    train_step = 0
    for ep in range(num_episodes):
        experiment_handler.log_training()
        # 5. perform scenario and save transitions
        ds_elem = train_ds[random.randint(0, train_size - 1)]
        p0, pk = ds_elem
        x0, y0, th0 = p0
        xk, yk, thk = pk
        xs, ys, ths = x0[tf.newaxis], y0[tf.newaxis], th0[tf.newaxis]
        xk, yk, thk = xk[tf.newaxis], yk[tf.newaxis], thk[tf.newaxis]
        ddy = tf.zeros_like(x0)[tf.newaxis]
        k = 0
        N = 5
        tau = 0.1
        gamma = 0.9
        # print(xs, ys, ths)
        p = []
        while True:
            # print(xs, ys, ths)
            params = actor(xs, ys, ths, ddy, xk, yk, thk)
            #params = tf.random_normal(tf.shape(params), params, tf.stack([0.1, 0.1, 0.05, 0.05], 0)[tf.newaxis])
            params = tf.random_normal(tf.shape(params), params, tf.stack([1.0, 1.0, 0.3, 0.3], 0)[tf.newaxis])
            px = params[:, 0]
            py = params[:, 0]
            pdy = params[:, 0]
            pddy = params[:, 0]
            px = tf.abs(px)
            pdy = tf.clip_by_value(pdy, -pi, pi)
            params = tf.stack([px, py, pdy, pddy], -1)
            p.append(params)
            # print(params)
            model_loss, invalid_loss, overshoot_loss, curvature_loss, \
            x_path, y_path, th_path, can_finish = plan_loss(params[:, :, tf.newaxis], env, xs, ys, ths, xk, yk, thk)
            # print("ML:", model_loss)
            # print("IL:", invalid_loss)
            # print("OL:", overshoot_loss)
            # print("CL:", curvature_loss)
            # _plot(x_path, y_path, th_path, env, ep, True)

            # TODO probably need change
            reward = -model_loss + 10 * float(can_finish)
            print(reward)
            if can_finish:
                print("FINISHED!!!!")

            xn, yn, thn = calculate_next_point(params, xs, ys, ths, ddy)
            xs = xn
            ys = yn
            ths = thn
            ddy = params[:, -1]
            # print(xn, yn, thn)

            episode_finished = k >= N or can_finish or invalid_loss.numpy() > 0.0

            replay_buffer.add((xs, ys, ths, ddy, xk, yk, thk), params, reward, (xn, yn, thn, ddy, xk, yk, thk),
                              episode_finished)

            if episode_finished:
                break
            k += 1

        if replay_buffer.count() > 128:
            # 6. if replay buffer full then update networks
            data = replay_buffer.get_batch(args.batch_size)
            x0 = tf.stack([x[0][0] for x in data], 0)
            y0 = tf.stack([x[0][1] for x in data], 0)
            th0 = tf.stack([x[0][2] for x in data], 0)
            ddy0 = tf.stack([x[0][3] for x in data], 0)
            xk = tf.stack([x[0][4] for x in data], 0)
            yk = tf.stack([x[0][5] for x in data], 0)
            thk = tf.stack([x[0][6] for x in data], 0)

            action = tf.concat([x[1] for x in data], 0)
            rewards = tf.stack([x[2] for x in data], 0)

            xn = tf.stack([x[3][0] for x in data], 0)
            yn = tf.stack([x[3][1] for x in data], 0)
            thn = tf.stack([x[3][2] for x in data], 0)
            ddyn = tf.stack([x[3][3] for x in data], 0)
            can_finish = tf.cast(tf.stack([x[4] for x in data], 0), dtype=tf.float32)

            with tf.GradientTape(persistent=True) as tape:
                next_action = target_actor(xn, yn, thn, ddyn, xk, yk, thk)[:, 0]
                pi_action = actor(x0, y0, th0, ddy0, xk, yk, thk)[:, 0]
                q = critic(x0, y0, th0, ddy0, xk, yk, thk, action)
                q_pi = critic(x0, y0, th0, ddy0, xk, yk, thk, pi_action)
                y = tf.stop_gradient(
                    rewards + gamma * (1 - can_finish) * target_critic(xn, yn, thn, ddyn, xk, yk, thk, next_action))

                #model_loss, invalid_loss, overshoot_loss, curvature_loss, \
                #x_path, y_path, th_path, can_finish = plan_loss(action[:1, :, tf.newaxis], env, x0[0], y0[0], th0[0], xk[0], yk[0], thk[0])
                #_plot(x_path, y_path, th_path, env, train_step, True)

                critic_loss = tf.keras.losses.mean_squared_error(y, q)
                actor_loss = -tf.reduce_mean(q_pi)
            critic_grads = tape.gradient(critic_loss, critic.trainable_variables)
            critic_optimizer.apply_gradients(zip(critic_grads, critic.trainable_variables),
                                             global_step=tf.train.get_or_create_global_step())
            actor_grads = tape.gradient(actor_loss, actor.trainable_variables)
            actor_optimizer.apply_gradients(zip(actor_grads, actor.trainable_variables),
                                            global_step=tf.train.get_or_create_global_step())
            # target_actor.trainable_variables = [tau * actor.trainable_variables[i] +
            #                                    (1. - tau) * target_actor.trainable_variables[i]
            #                                    for i in range(len(actor.trainable_variables))]
            # target_critic.trainable_variables = [tau * critic.trainable_variables[i] +
            #                                    (1. - tau) * target_critic.trainable_variables[i]
            #                               violation_level = tf.reduce_sum(in_obstacle, -1)         for i in range(len(actor.trainable_variables))]

            for i in range(len(actor.trainable_variables)):
                tf.assign(target_actor.trainable_variables[i],
                          tau * actor.trainable_variables[i] + (1. - tau) * target_actor.trainable_variables[i])
            for i in range(len(critic.trainable_variables)):
                tf.assign(target_critic.trainable_variables[i],
                          tau * critic.trainable_variables[i] + (1. - tau) * target_critic.trainable_variables[i])

            with tfc.summary.record_summaries_every_n_global_steps(args.log_interval, train_step):
                tfc.summary.scalar('metrics/critic_loss', critic_loss[0], step=train_step)
                tfc.summary.scalar('metrics/actor_loss', actor_loss, step=train_step)
                tfc.summary.scalar('metrics/y', y, step=train_step)
                tfc.summary.scalar('metrics/q', q, step=train_step)
                # tfc.summary.scalar('metrics/overshoot_loss', overshoot_loss, step=train_step)
                # tfc.summary.scalar('metrics/curvature_loss', curvature_loss, step=train_step)
                # tfc.summary.scalar('metrics/reg_loss', reg_loss, step=train_step)
                # tfc.summary.scalar('metrics/good_paths', s, step=train_step)
                # tfc.summary.scalar('training/eta', eta, step=train_step)
            train_step += 1

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
