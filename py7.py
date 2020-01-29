import numpy as np
import tensorflow as tf


def one_hot_matrix(labels, Con):
    """
     Creates a matrix where the i-th row corresponds to the ith class number and the jth column
     corresponds to the jth training example. So if example j had a label i. Then entry (i,j) 
     will be 1. 
    """
    C = tf.constant(Con, name='C')
    one_hot_matrix = tf.one_hot(indices=labels, depth=C, axis=0)
    with tf.Session() as sess:
        one_hot = sess.run(one_hot_matrix)

    sess.close()
    return one_hot


labels = np.array([1, 2, 0, 1])
one_hot = one_hot_matrix(labels, Con=3)
print(one_hot)
