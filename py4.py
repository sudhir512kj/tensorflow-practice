import numpy as np
import tensorflow as tf


def linear_function():
    # Initializes X to be a random tensor of shape (4,1)
    X = np.random.randn(4, 1)
    # Initializes W to be a random tensor of shape (5,4)
    W = np.random.randn(5, 4)
    # Initializes b to be a random tensor of shape (5,1)
    b = np.random.randn(5, 1)

    Y = tf.add(tf.matmul(W, X), b)  # computation graph

    sess = tf.Session()
    result = sess.run(Y)  # evaluating the computation graph

    sess.close()  # closing the tf session
    return result


print(linear_function())
