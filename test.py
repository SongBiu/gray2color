import tensorflow as tf
import numpy as np

a = tf.placeholder(dtype=tf.float32, shape=[2,2,2])
b = tf.placeholder(dtype=tf.float32, shape=[2,2,2])

c = tf.losses.mean_squared_error(a,b)

x1 = np.array([i for i in range(8)]).reshape((2,2,2)).astype(np.float32)
x2 = np.array([i*2 for i in range(8)]).reshape((2,2,2)).astype(np.float32)

sess = tf.Session()
d = sess.run(c, feed_dict={a:x1, b:x2})
print(d)