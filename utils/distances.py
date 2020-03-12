#!/usr/bin/python
import tensorflow as tf
import numpy as np


def dist2vert(v, q):
    v = tf.expand_dims(v, 1)
    q = tf.expand_dims(q, 2)
    d = euclid(v, q)
    d = tf.reduce_min(d, 2)
    return d


def euclid(a, b=None):
    if b is None:
        return tf.sqrt(tf.reduce_sum(a ** 2, -1))
    return tf.sqrt(tf.reduce_sum((a - b) ** 2, -1))


def cross_product(a, b):
    return a[:, :, :, :, :, 0] * b[:, :, :, :, :, 1] - b[:, :, :, :, :, 0] * a[:, :, :, :, :, 1]


def point2edge(verts, query_points):
    first_point_coords = verts
    second_point_coords = tf.roll(verts, -1, -2)
    edge_vector = second_point_coords - first_point_coords
    edge_vector = edge_vector[:, tf.newaxis, tf.newaxis]
    query_points_in_v1 = query_points[:, :, :, tf.newaxis, tf.newaxis] - first_point_coords[:, tf.newaxis, tf.newaxis]
    p = tf.reduce_sum(edge_vector * query_points_in_v1, -1)
    cross = cross_product(edge_vector, query_points_in_v1)
    inside = tf.logical_or(tf.reduce_all(cross > 0, -1), tf.reduce_all(cross < 0, -1))
    inside = tf.reduce_any(inside, -1)
    t = tf.reduce_sum(edge_vector * edge_vector, -1)
    w = p / (t + 1e-8)
    w = tf.where(w <= 0, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    w = tf.where(w >= 1, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    p = edge_vector * tf.expand_dims(w, -1) \
        + first_point_coords[:, tf.newaxis, tf.newaxis]  # calcualte point on the edge
    return p - query_points[:, :, :, tf.newaxis, tf.newaxis], inside


def point2vert(verts, query_points):
    return verts[:, tf.newaxis, tf.newaxis] - query_points[:, :, :, tf.newaxis, tf.newaxis]


def dist(verts, query_points):
    """

    :param verts: (N, V, 4, 2)
    :param query_points: (N, S, P, 2)
    :return:
    """
    p2e, inside = point2edge(verts, query_points)
    p2v = point2vert(verts, query_points)
    p2e = tf.reduce_sum(tf.abs(p2e), axis=-1)
    p2v = euclid(p2v)
    dists = tf.concat([p2e, p2v], -1)
    dists = tf.reduce_min(dists, (-2, -1))
    dists = tf.where(inside, tf.zeros_like(dists), dists)
    return dists

import matplotlib.pyplot as plt


def if_inside(verts, query_points):
    first_point_coords = verts
    second_point_coords = tf.roll(verts, -1, -2)
    edge_vector = second_point_coords - first_point_coords
    edge_vector = edge_vector[:, tf.newaxis, tf.newaxis]
    query_points_in_v1 = query_points[:, :, :, tf.newaxis, tf.newaxis] - first_point_coords[:, tf.newaxis, tf.newaxis]
    p = tf.reduce_sum(edge_vector * query_points_in_v1, -1)
    cross = cross_product(edge_vector, query_points_in_v1)
    inside = tf.logical_or(tf.reduce_all(cross > 0, -1), tf.reduce_all(cross < 0, -1))
    inside = tf.reduce_any(inside, -1)
    return inside


def integral(free_space, path):
    segment_centers = (path[:, 1:] + path[:, :-1]) / 2.
    segments_direction = path[:, 1:] - path[:, :-1]
    segments_lengths = tf.linalg.norm(segments_direction, axis=-1, keep_dims=True)
    segments_perpendicular_direction = tf.reverse(segments_direction, [-1]) / (segments_lengths + 1e-8)
    perp_line = segments_perpendicular_direction[:, tf.newaxis] * tf.linspace(0.0, 1.0, 2)[tf.newaxis, :, tf.newaxis,
                                                                  tf.newaxis, tf.newaxis]
    p = segment_centers[:, tf.newaxis] + perp_line
    x1 = p[:, 0, :, :, 0]
    x2 = p[:, 1, :, :, 0]
    y1 = p[:, 0, :, :, 1]
    y2 = p[:, 1, :, :, 1]
    Ap = y2 - y1
    Bp = x1 - x2
    Cp = x2 * y1 - x1 * y2

    first_point_coords = free_space
    second_point_coords = tf.roll(free_space, -1, -2)
    free_space_segments = tf.stack([first_point_coords, second_point_coords], -2)

    Ap = Ap[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    Bp = Bp[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    Cp = Cp[:, :, :, tf.newaxis, tf.newaxis, tf.newaxis]
    free_space_segments = free_space_segments[:, tf.newaxis, tf.newaxis]
    xs = free_space_segments[:, :, :, :, :, :, 0]
    ys = free_space_segments[:, :, :, :, :, :, 1]
    if_on_the_other_sides = Ap * xs + Bp * ys + Cp
    if_on_the_other_sides = tf.less(tf.reduce_prod(if_on_the_other_sides, -1), 0)

    x1 = xs[:, :, :, :, :, 0]
    y1 = ys[:, :, :, :, :, 0]
    x2 = xs[:, :, :, :, :, 1]
    y2 = ys[:, :, :, :, :, 1]
    As = y2 - y1
    Bs = x1 - x2
    Cs = x2 * y1 - x1 * y2

    segment_centers = segment_centers[:, :, :, tf.newaxis, tf.newaxis]
    xsc = segment_centers[:, :, :, :, :, 0]
    ysc = segment_centers[:, :, :, :, :, 1]
    inside = As * xsc + Bs * ysc + Cs
    inside = inside < 0
    inside = tf.reduce_all(inside, -1)

    D = tf.squeeze(Ap, -1) * Bs - As * tf.squeeze(Bp, -1)
    D = tf.where(if_on_the_other_sides, D, 1e-8 * tf.ones_like(D))
    xc = (tf.squeeze(Bp, -1) - Bs) / D
    yc = -(tf.squeeze(Ap, -1) - As) / D

    xyc = tf.stack([xc, yc], -1)
    dist = tf.linalg.norm(xyc - segment_centers, axis=-1)
    dist = dist * tf.cast(if_on_the_other_sides, tf.float32)
    dist = tf.where(if_on_the_other_sides, dist, 1e10 * tf.ones_like(dist))

    dist = tf.reduce_min(dist, -1)
    dist = tf.where(inside, tf.zeros_like(dist), dist)
    dist = tf.reduce_min(dist, -1)

    penetration = dist * segments_lengths[:, :, :, 0]
    penetration = tf.reduce_mean(penetration, -1)

    return tf.reduce_sum(penetration, -1)


def dist_perpendicular(free_space, p):
    """

    :param free_space: (N, V, 4, 2)
    :param p: (N, S, P, 2)
    :return:
    """

    p = tf.transpose(p, (0, 2, 1, 3))
    quads_len = free_space.shape[1]
    p_0 = p[:, :, :-1]
    p_1 = p[:, :, 1:]
    pts_len = tf.shape(p_0)[1]
    seq_len = tf.shape(p_0)[2]
    p_0_x = p_0[..., 0]
    p_0_y = p_0[..., 1]
    p_1_x = p_1[..., 0]
    p_1_y = p_1[..., 1]
    A = p_1_y - p_0_y
    B = p_0_x - p_1_x
    C = p_1_x * p_0_y - p_0_x * p_1_y
    s = (p_0 + p_1) / 2

    p = tf.transpose(p, (0, 2, 1, 3))
    inside = if_inside(free_space, p)
    inside = tf.transpose(inside, (0, 2, 1))[..., :-1]

    A2_B2 = - B / (A + 1e-10)
    AB2 = (tf.zeros_like(A), tf.zeros_like(B))
    AB2 = tf.where((tf.equal(B, 0.0), tf.equal(B, 0.0)), (tf.zeros_like(A), tf.ones_like(B)), AB2)
    AB2 = tf.where((tf.equal(A, 0.0), tf.equal(A, 0.0)), (tf.ones_like(A), tf.zeros_like(B)), AB2)
    A_or_B = tf.logical_or(tf.equal(B, 0.0), tf.equal(A, 0.0))
    AB2 = tf.where((A_or_B, A_or_B), AB2, (A2_B2, tf.ones_like(A)))
    AB2 = tf.transpose(AB2, (1, 2, 3, 0))
    C2 = - AB2[..., 0] * s[..., 0] - AB2[..., 1] * s[..., 1]

    AB2 = tf.tile(AB2[:, tf.newaxis, tf.newaxis], (1, quads_len, 4, 1, 1, 1))
    C2 = tf.tile(C2[:, tf.newaxis, tf.newaxis], (1, quads_len, 4, 1, 1))

    free_space = tf.tile(free_space[:, :, :, tf.newaxis, tf.newaxis], (1, 1, 1, pts_len, seq_len, 1))
    fs_0 = free_space
    fs_1 = tf.roll(free_space, 1, axis=2)
    fs_0_x = fs_0[..., 0]
    fs_0_y = fs_0[..., 1]
    fs_1_x = fs_1[..., 0]
    fs_1_y = fs_1[..., 1]
    fs_A = fs_1_y - fs_0_y
    fs_B = fs_0_x - fs_1_x
    fs_AB = tf.stack([fs_A, fs_B], -1)
    fs_C = fs_1_x * fs_0_y - fs_0_x * fs_1_y

    Cs = tf.stack([C2, fs_C], -1)
    M = tf.stack([AB2, fs_AB], -2)
    det = tf.linalg.det(M)
    det = tf.tile(det[..., tf.newaxis, tf.newaxis], (1, 1, 1, 1, 1, 2, 2))
    M = tf.where(det < 1e-4, M + tf.eye(2, dtype=tf.float32)[tf.newaxis] * 1e-4, M)
    pp = tf.linalg.solve(M, -Cs[..., tf.newaxis])
    #pp = tf.linalg.solve(M + tf.eye(2, dtype=tf.float32)[tf.newaxis] * 1e-8, -Cs[..., tf.newaxis])
    pp = pp[..., 0]

    # CHECK IF BETWEEN ENDS OF SEGMENT
    edge = fs_1 - fs_0
    pp_in_fs_0 = pp - fs_0

    g = tf.reduce_sum(edge * pp_in_fs_0, -1)
    t = tf.reduce_sum(edge * edge, -1)
    w = g / (t + 1e-8)
    w = tf.where(tf.logical_and(w >= 0, w <= 1), tf.ones_like(w),
                 1e10 * tf.ones_like(w))  # ignore points outside of edge

    pp = pp * w[..., tf.newaxis]
    dist = tf.linalg.norm(pp - s[:, tf.newaxis, tf.newaxis], axis=-1)
    dist = tf.reduce_min(dist, 2)  # min from edges
    dist = tf.reduce_min(dist, 1)  # min from quads

    dist = tf.where(inside, tf.zeros_like(dist), dist)
    dist = tf.where(tf.greater(dist, 1e5), tf.zeros_like(dist), dist)
    dist = tf.transpose(dist, (0, 2, 1))
    return dist
