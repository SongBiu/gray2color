import os
import numpy as np
import tensorflow as tf
from PIL import Image
import network
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

if sys.argv[2] == 'train':
	name = "../64/gray/%s.png" % sys.argv[1]
else:
	name = "../64/test/gray/%s.png" % sys.argv[1]

x = np.zeros((1, 64, 64, 1))
image = np.array(Image.open(name))
x[0, :, :, 0] = image[:, :]

image_gray = tf.placeholder(dtype=tf.float32, shape=[-1, image_size, image_size, 1])
image_color = tf.placeholder(dtype=tf.float32, shape=[-1, image_size, image_size, 3])
net_g, _ = network.network_g(image_gray=image_gray, is_train=False, reuse=False)

d_input_real = tf.concat([image_gray, image_color], axis=3)
d_input_fake = tf.concat([image_gray, net_g*255], axis=3)
logits_real, _ = network.network_d(image_input=d_input_real, is_train=False, reuse=False)
logits_fake, _ = network.network_d(image_input=d_input_fake, is_train=False, reuse=True)

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "ckp")

x_out = sess.run(net_g*255, feed_dict={image_gray: x})
# print mat.shape
image = np.zeros((64, 64, 3))
image[:, :, :] = x_out[0, :, : ,:]
image = image.astype(np.int32).astype(np.uint8)
out = Image.fromarray(image)
out.save('tmp-'+sys.argv[1]+'.png')