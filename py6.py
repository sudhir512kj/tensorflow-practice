import numpy as np
import tensorflow as tf


def cost(logits, labels):
    """
    Computes the cost using the sigmoid cross entropy

    Arguments:
    logits -- vector containing z, output of the last linear unit (before the final sigmoid activation) 
    labels -- vector of labels y (1 or 0) 
    """
    z = tf.placeholder(tf.float32, name='z')
    y = tf.placeholder(tf.float32, name='y')

    # the loss function -
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=z, labels=y)

    # running the session -
    with tf.Session() as sess:
        cost = sess.run(cost, feed_dict={z: logits, y: labels})

    sess.close()
    return cost


logits = np.array([0.29, 0.34, 0.75, 0.88])
cost = cost(logits, np.array([0, 0, 1, 1]))
print(cost)
