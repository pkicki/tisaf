import os
import numpy as np
from tensorflow.contrib.opt import ScipyOptimizerInterface

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import tensorflow.contrib as tfc
import matplotlib.pyplot as plt

tf.enable_eager_execution()


class M(tf.keras.Model):
    def __init__(self):
        super(M, self).__init__()
        self.x = tf.Variable(10.0, trainable=True, name='x')
        self.y = tf.Variable(-8.0, trainable=True, name='y')

    def call(self, input):
        return self.x, self.y
        # return tf.concat([self.x, self.y], 0)


A = 1
B = 150
#B = 1


def loss_fcn(x, y):
    return A * (x - 2) ** 2 + B * (y - 2) ** 2


def loss_fcn_goal(x, y, a, b):
    return (x - a) ** 2 + (y - b) ** 2

tf.random.set_random_seed(444)

eta = 1e-1
# eta = 1e0
optimizer = tf.train.AdamOptimizer(eta)
opt = tf.train.GradientDescentOptimizer(1e-1)
#opt = tf.train.AdamOptimizer(1e-1)
model = M()

EPS = 1e-2

mine = False
#mine = True
xs = []
ys = []
random_calls = 0
#for i in range(100):
for i in range(1000):
    # for i in range(1000):
    # for i in range(1):
    if mine:
        ## my model
        x, y = model(i)
        a = 0.0
        b = 0.0
        k = 0
        while a >= b:
            xr = tf.random_normal([30], x, stddev=1.0)
            yr = tf.random_normal([30], y, stddev=1.0)
            res = loss_fcn(xr, yr)
            a = tf.reduce_min(res).numpy()
            b = loss_fcn(x, y).numpy()
            random_calls += 1

        if b < EPS:
            print(i)
            print(random_calls)
            break

        idx = tf.argmin(res)
        xc = xr[idx]
        yc = yr[idx]
        print(x, y)
        print(xc, yc)
        print("MIN VAL:", a)
        print("ACT VAL:", b)
        for i in range(3):
            with tf.GradientTape() as tape:
                x, y = model(i)
                model_loss = loss_fcn_goal(x, y, xc, yc)
                # print(model_loss)
                print(loss_fcn(x, y))
                total_loss = model_loss

            # 5.1.2 Take gradients (if necessary apply regularization like clipping),
            grads = tape.gradient(total_loss, model.trainable_variables)
            # print(grads)
            # grads = [tf.clip_by_value(g, -1., 1.) for g in grads]
            #grads = [tf.clip_by_norm(g, 1.) for g in grads]
            opt.apply_gradients(zip(grads, model.trainable_variables),
                                global_step=tf.train.get_or_create_global_step())
        x, y = model(i)
        xs.append(x.numpy())
        ys.append(y.numpy())
    else:
        x, y = model(i)
        print(x, y)
        with tf.GradientTape() as tape:
            model_loss = loss_fcn(x, y)

            if model_loss < EPS:
                print(i)
                break

            print(model_loss)
            xs.append(x.numpy())
            ys.append(y.numpy())
            total_loss = model_loss

        # 5.1.2 Take gradients (if necessary apply regularization like clipping),
        grads = tape.gradient(total_loss, model.trainable_variables)
        print(grads)
        # grads = [tf.clip_by_value(g, -1., 1.) for g in grads]
        #grads = [tf.clip_by_norm(g, 1.) for g in grads]
        optimizer.apply_gradients(zip(grads, model.trainable_variables),
                                  global_step=tf.train.get_or_create_global_step())
plt.plot(xs, ys)
plt.show()
