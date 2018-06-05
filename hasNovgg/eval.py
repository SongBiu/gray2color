import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from PIL import Image
from network import network_g
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
if sys.argv[2] == 'train':
	name = "../64/gray/%s.png" % sys.argv[1]
else:
	name = "../64/test/gray/%s.png" % sys.argv[1]
x = np.zeros((1, 64, 64, 1))
image = np.array(Image.open(name))
x[0, :, :, 0] = image[:, :]
x_input = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1])
y = network_g(x_input, False, False)
sess = tf.Session()
tl.layers.initialize_global_variables(sess)
tl.files.load_and_assign_npz(sess=sess, name="g_3.npz", network=y)
mat = sess.run(y.outputs*255, feed_dict={x_input: x})
# print mat.shape
image = np.zeros((64, 64, 3))
image[:, :, :] = mat[0, :, : ,:]
image = image.astype(np.uint8)
out = Image.fromarray(image)
out.save('tmp.png')