import tensorflow as tf  # Creating the computation graph  -
a = tf.constant(2)  # tensor "a"
b = tf.constant(10)  # tensor "b"
c = tf.multiply(a, b)  # tensor "c"# Run the computation graph
sess = tf.Session()
print(sess.run(c))


x = tf.placeholder(tf.int64, name='x')  # a placeholder tensor
print(sess.run(10 * x, feed_dict={x: 2}))  # running the graph
sess.close()
