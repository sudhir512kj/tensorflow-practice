import numpy as np
import tensorflow as tf


def sigmoid(z):
    x = tf.placeholder(tf.float32, name='x')  # a placeholder tensor (variable)
    sigmoid = tf.sigmoid(x)  # compting sigmoid function of "x"

    with tf.Session() as sess:
        # runnig the computation graph (sigmoid) using a feed_dict
        result = sess.run(sigmoid, feed_dict={x: z})
    return result


print(sigmoid(1))
print(sigmoid(10))
print(sigmoid(100))
