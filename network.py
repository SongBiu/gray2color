import tensorflow as tf
import time


def convWithBN(inputs, channels, shape, is_train, nameid, activation=tf.nn.leaky_relu):

	conv = tf.layers.conv2d(inputs=inputs, filters=channels, kernel_size=shape[0], 
		strides=shape[1], padding="same", activation=activation, name="conv"+nameid)
	
	conv = tf.layers.batch_normalization(inputs=conv, trainable=True, training=is_train,
		name="bnAfterConv"+nameid)

	return conv


def deconvWithBN(inputs, channels, shape, is_train, nameid, activation=tf.nn.leaky_relu):
	
	deconv = tf.layers.conv2d_transpose(inputs=inputs, filters=channels, kernel_size=shape[0], 
		strides=shape[1], padding="same", activation=activation, name="deconv"+nameid)

	deconv = tf.layers.batch_normalization(inputs=deconv, trainable=True, training=is_train,
		name="bnAfterDeconv"+nameid)

	return deconv


def network_g(image_gray, is_train, reuse=None):

	with tf.variable_scope('network_g', reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)
		net = tl.layers.InputLayer(inputs=image_gray, name="input_layer")
		net = tl.layers.Conv2d(net, n_filter=16, filter_size=(3, 3), strides=(2, 2), name="pre_conv_1")
		net = tl.layers.Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="pre_conv_2")

		skip = net
		for i in range(12):
			nn = tl.layers.Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="block%d_conv1" % i)
			nn = tl.layers.BatchNormLayer(nn, is_train=is_train, act=tf.nn.leaky_relu, name="block%d_bn_1" % i)
			nn = tl.layers.Conv2d(nn, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="block%d_conv2" % i)
			nn = tl.layers.BatchNormLayer(nn, is_train=is_train, name="block%d_bn_2" % i)
			nn = tl.layers.ElementwiseLayer([nn, net], combine_fn=tf.add, name="block%d_add" % i)
			net = nn

		net = tl.layers.Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(1, 1), name='after_c')
		net = tl.layers.BatchNormLayer(net, is_train=is_train, name='after_bn')
		net = tl.layers.ElementwiseLayer([skip, net], combine_fn=tf.add, name='after_add')

		for i in range(2):
			net = tl.layers.Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(1, 1), name='subpixel%d_conv' % i)
			net = tl.layers.SubpixelConv2d(net, scale=2, act=tf.nn.relu, name='subpixel%d_sub' % i)

		net = tl.layers.Conv2d(net, n_filter=3, filter_size=(1, 1), strides=(1, 1), act=tf.tanh, name="out")
		return net
		

def network_d(image_input, is_train, keep_prob, reuse=None):

	with tf.variable_scope('network_d', reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)
		net = tl.layers.InputLayer(inputs=image_input, name="input_layer")
		net = tl.layers.Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="pre/conv")
		net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), name="pre/pooling")

		skip = net
		for i in range(7):
			nn = tl.layers.Conv2d(net, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="block%d_conv1" % i)
			nn = tl.layers.BatchNormLayer(nn, is_train=is_train, act=tf.nn.leaky_relu, name="block%d_bn_1" % i)
			nn = tl.layers.Conv2d(nn, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="block%d_conv2" % i)
			nn = tl.layers.BatchNormLayer(nn, is_train=is_train, name="block%d_bn_2" % i)
			nn = tl.layers.ElementwiseLayer([nn, net], combine_fn=tf.add, name="block%d_add" % i)
			net = nn

		net = tl.layers.Conv2d(net=net, n_filter=64, filter_size=(3, 3), strides=(1, 1), name='after_c')
		net = tl.layers.BatchNormLayer(layer=net, is_train=is_train, name='after_bn')
		net = tl.layers.ElementwiseLayer(layer=[skip, net], combine_fn=tf.add, name='after_add')

		net = tl.layers.FlattenLayer(net, name="flatten")
		net = tl.layers.DenseLayer(net, n_units=64, act=tf.nn.relu, name="dense64")
		net = tl.layers.DenseLayer(net, n_units=512, act=tf.nn.relu, name="dense512")
		pro = tl.layers.DenseLayer(net, n_units=1, name="dense1/p")
		return net, pro

