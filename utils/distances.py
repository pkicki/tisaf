#!/usr/bin/python
import tensorflow as tf


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
    w = p / t
    w = tf.where(w <= 0, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    w = tf.where(w >= 1, 1e10 * tf.ones_like(w), w)  # ignore points outside of edge
    p = edge_vector * tf.expand_dims(w, -1)\
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