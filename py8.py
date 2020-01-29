import numpy as np
import tensorflow as tf


def zeros_ones(shape):
    zeros = tf.zeros(shape)  # "zeros" tensor
    ones = tf.ones(shape)  # "ones" tensor

    with tf.Session() as sess:
        zeros = sess.run(zeros)
        ones = sess.run(ones)

    sess.close()
    return zeros, ones


print(zeros_ones(3))
