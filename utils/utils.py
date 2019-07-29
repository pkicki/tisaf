import tensorflow as tf

class Pose2D:
    def __init__(self, x, y, fi):
        self.x = x
        self.y = y
        self.fi = fi


class SingleWagonBus(Pose2D):
    def __init__(self, x, y, fi, beta):
        super().__init__(x, y, fi)
        self.beta = beta


def Rot(fi):
    c = tf.cos(fi)
    s = tf.sin(fi)
    L = tf.stack([c, s], -1)
    R = tf.stack([-s, c], -1)
    return tf.stack([L, R], -1)


def angleFromRot(R):
    return tf.atan2(R[:, 1, 0], R[:, 0, 0])


