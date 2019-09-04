import numpy as np
import tensorflow as tf
from dataset import scenarios
from utils.distances import integral

tf.enable_eager_execution()

train_ds, train_size, free_space = scenarios.planning_dataset("../../dummy_data/train/prosta/")
bs = 1
# free_space = tf.tile(free_space, [bs, 6, 1, 1])
free_space = tf.tile(free_space, [bs, 1, 1, 1])
path = np.array([[7.0, 1.0], [7.0, 3.0], [6.0, 4.0]], dtype=np.float32)
# path = np.array([[3.0, 1.0], [3.0, 3.0], [2.0, 4.0]], dtype=np.float32)
path = path[np.newaxis, :, np.newaxis]
# path = tf.tile(path, [bs, 1, 7, 1])
path = tf.tile(path, [bs, 1, 1, 1])

path = tf.Variable(path, trainable=True)

with tf.GradientTape() as tape:
    r = integral(free_space, path)
    print(r)

grad = tape.gradient(r, path)
print(grad)
