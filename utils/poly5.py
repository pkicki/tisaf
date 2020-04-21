import tensorflow as tf


def params(q0, q1):
    x0, y0, dy0, ddy0 = q0
    x1, y1, dy1, ddy1 = q1
    ones = tf.ones_like(x0, dtype=tf.float32)
    zeros = tf.zeros_like(x0, dtype=tf.float32)

    X0 = tf.stack([x0 ** 5, x0 ** 4, x0 ** 3, x0 ** 2, x0, ones], -1)
    dX0 = tf.stack([5 * x0 ** 4, 4 * x0 ** 3, 3 * x0 ** 2, 2 * x0, ones, zeros], -1)
    ddX0 = tf.stack([20 * x0 ** 3, 12 * x0 ** 2, 6 * x0, 2 * ones, zeros, zeros], -1)

    X1 = tf.stack([x1 ** 5, x1 ** 4, x1 ** 3, x1 ** 2, x1, ones], -1)
    dX1 = tf.stack([5 * x1 ** 4, 4 * x1 ** 3, 3 * x1 ** 2, 2 * x1, ones, zeros], -1)
    ddX1 = tf.stack([20 * x1 ** 3, 12 * x1 ** 2, 6 * x1, 2 * ones, zeros, zeros], -1)

    A = tf.stack([X0, dX0, ddX0, X1, dX1, ddX1], -2)
    b = tf.stack([y0, dy0, ddy0, y1, dy1, ddy1], -1)
    # print(A)
    h = tf.linalg.solve(A, b[..., tf.newaxis])
    return h


def params_xy(q0, q1, m):
    x0, y0, dy0, ddy0 = tf.unstack(q0, axis=-1)
    x1, y1, dy1, ddy1 = tf.unstack(q1, axis=-1)
    m_dy0, m_ddy0, m_dy1, m_ddy1 = tf.unstack(m, axis=-1)

    ones = tf.ones_like(x0, dtype=tf.float32)
    zeros = tf.zeros_like(x0, dtype=tf.float32)

    T1 = tf.stack([zeros, zeros, zeros, zeros, zeros, ones])
    T2 = tf.stack([zeros, zeros, zeros, zeros, ones, -5 * ones])
    T3 = tf.stack([zeros, zeros, zeros, 2 * ones, -8 * ones, 20 * ones])
    T4 = tf.stack([ones, zeros, zeros, zeros, zeros, zeros])
    T5 = tf.stack([5 * ones, -ones, zeros, zeros, zeros, zeros])
    T6 = tf.stack([20 * ones, -8 * ones, 2 * ones, zeros, zeros, zeros])
    T = tf.stack([T1, T2, T3, T4, T5, T6], -2)
    T = tf.transpose(T, (2, 1, 0))
    z = tf.stack([x0, m_dy0, m_ddy0, x1, m_dy1, m_ddy1], -1)
    a = tf.linalg.solve(T, z[..., tf.newaxis])

    T1 = tf.stack([zeros, zeros, zeros, zeros, zeros, ones])
    T2 = tf.stack([zeros, zeros, zeros, zeros, ones, -5 * ones])
    T3 = tf.stack([zeros, zeros, zeros, 2 * ones, -8 * ones, 20 * ones])
    T4 = tf.stack([ones, zeros, zeros, zeros, zeros, zeros])
    T5 = tf.stack([5 * ones, -ones, zeros, zeros, zeros, zeros])
    T6 = tf.stack([20 * ones, -8 * ones, 2 * ones, zeros, zeros, zeros])
    T = tf.stack([T1, T2, T3, T4, T5, T6], -2)
    T = tf.transpose(T, (2, 1, 0))
    dy_dt_0 = dy0 * m_dy0
    d2y_dt2_0 = (ddy0 - dy_dt_0 / m_ddy0) * m_dy0 ** 2
    dy_dt_1 = dy1 * m_dy1
    d2y_dt2_1 = (ddy1 - dy_dt_1 / m_ddy1) * m_dy1 ** 2
    z = tf.stack([y0, dy_dt_0, d2y_dt2_0, y1, dy_dt_1, d2y_dt2_1], -1)
    b = tf.linalg.solve(T, z[..., tf.newaxis])
    return tf.transpose(a, (0, 2, 1)), tf.transpose(b, (0, 2, 1))


def curvature(p, s):
    ddt = tf.stack(DDT(s), 0)
    dt = tf.stack(DT(s), 0)
    t = tf.stack(T(s), 0)
    x = tf.squeeze(p[0] @ t, 1)
    y = tf.squeeze(p[1] @ t, 1)
    dx = tf.squeeze(p[0] @ dt, 1)
    dy = tf.squeeze(p[1] @ dt, 1)
    ddx = tf.squeeze(p[0] @ ddt, 1)
    ddy = tf.squeeze(p[1] @ ddt, 1)
    curv = (ddy * dx - ddx * dy) / (dx ** 2 + dy ** 2) ** (3 / 2)
    return curv, dx, dy, x, y


def DY(p, x):
    ones = tf.ones_like(x)
    zeros = tf.zeros_like(x)
    dX = tf.stack([5 * x ** 4, 4 * x ** 3, 3 * x ** 2, 2 * x, ones, zeros], -1)
    dY = tf.squeeze(dX @ p, -1)
    return dY


def T(s):
    return [s ** 5,
            s ** 4 * (1 - s),
            s ** 3 * (1 - s) ** 2,
            s ** 2 * (1 - s) ** 3,
            s * (1 - s) ** 4,
            (1 - s) ** 5]


def DT(s):
    return [5 * s ** 4,
            4 * s ** 3 * (1 - s) - s ** 4,
            3 * s ** 2 * (1 - s) ** 2 - 2 * s ** 3 * (1 - s),
            2 * s * (1 - s) ** 3 - 3 * s ** 2 * (1 - s) ** 2,
            (1 - s) ** 4 - 4 * s * (1 - s) ** 3,
            -5 * (1 - s) ** 4]


def DDT(s):
    return [20 * s ** 3,
            12 * s ** 2 - 20 * s ** 3,
            8 * s ** 3 - 12 * s ** 2 + 6 * s,
            2 * (1 - s) ** 3 - 12 * s * (s - 1) ** 2 + 6 * s ** 2 * (1 - s),
            -8 * (1 - s) ** 3 + 12 * s * (1 - s) ** 2,
            +20 * (1 - s) ** 3]
