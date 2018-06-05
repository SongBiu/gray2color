import tensorlayer as tl
import tensorflow as tf
import time


def network_g(image_gray, reuse, is_train):
	with tf.variable_scope('network_g', reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)
		input_layer = tl.layers.InputLayer(inputs=image_gray, name="input_layer")

		# 64*64*16
		conv1 = tl.layers.Conv2d(input_layer, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="conv1_1")
		conv1 = tl.layers.BatchNormLayer(conv1, is_train=is_train, act=tf.nn.leaky_relu, name="1_bn_1")
		conv1 = tl.layers.Conv2d(conv1, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="conv1_2")
		conv1 = tl.layers.BatchNormLayer(conv1, is_train=is_train, act=tf.nn.leaky_relu, name="1_bn_2")

		# 32*32*32
		conv2 = tl.layers.Conv2d(conv1, n_filter=32, filter_size=(3, 3), strides=(2, 2), name="conv2_1")
		conv2 = tl.layers.BatchNormLayer(conv2, is_train=is_train, act=tf.nn.leaky_relu, name="2_bn_1")
		conv2 = tl.layers.Conv2d(conv2, n_filter=32, filter_size=(3, 3), strides=(1, 1), name="conv2_2")
		conv2 = tl.layers.BatchNormLayer(conv2, is_train=is_train, act=tf.nn.leaky_relu, name="2_bn_2")
		
		# 16*16*64
		conv3 = tl.layers.Conv2d(conv2, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="conv3_1")
		conv3 = tl.layers.BatchNormLayer(conv3, is_train=is_train, act=tf.nn.leaky_relu, name="3_bn_1")
		conv3 = tl.layers.Conv2d(conv3, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="conv3_2")
		conv3 = tl.layers.BatchNormLayer(conv3, is_train=is_train, act=tf.nn.leaky_relu, name="3_bn_2")

		# 8*8*128
		conv4 = tl.layers.Conv2d(conv3, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="conv4_1")
		conv4 = tl.layers.BatchNormLayer(conv4, is_train=is_train, act=tf.nn.leaky_relu, name="4_bn_1")
		conv4 = tl.layers.Conv2d(conv4, n_filter=128, filter_size=(3, 3), strides=(1, 1), name="conv4_2")
		conv4 = tl.layers.BatchNormLayer(conv4, is_train=is_train, act=tf.nn.leaky_relu, name="4_bn_2")
	   
		# 4*4*256
		conv5 = tl.layers.Conv2d(conv4, n_filter=256, filter_size=(3, 3), strides=(2, 2), name="conv5_1")
		conv5 = tl.layers.BatchNormLayer(conv5, is_train=is_train, act=tf.nn.leaky_relu, name="5_bn_1")
		conv5 = tl.layers.Conv2d(conv5, n_filter=256, filter_size=(3, 3), strides=(1, 1), name="conv5_2")
		conv5 = tl.layers.BatchNormLayer(conv5, is_train=is_train, act=tf.nn.leaky_relu, name="5_bn_2")
		
		'''
		architecture of deconv-layer:
			deconv(upsample) => 
			concat(with corresponding conv) => 
			conv(1*1 conv)
		'''
		# 8*8*128
		deconv4 = tl.layers.DeConv2d(conv5, n_filter=128, filter_size=(3, 3), strides=(2, 2), name="deconv4")
		deconv4 = tl.layers.BatchNormLayer(deconv4, is_train=is_train, act=tf.nn.leaky_relu, name="d_1_bn_1")
		deconv4 = tl.layers.ConcatLayer([conv4, deconv4], concat_dim=3, name="concat_1")
		deconv4 = tl.layers.Conv2d(deconv4, n_filter=128, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv4")
		deconv4 = tl.layers.BatchNormLayer(deconv4, is_train=is_train, act=tf.nn.leaky_relu, name="d_1_bn_2")
		
		# 16*16*64
		deconv3 = tl.layers.DeConv2d(deconv4, n_filter=64, filter_size=(3, 3), strides=(2, 2), name="deconv3")
		deconv3 = tl.layers.BatchNormLayer(deconv3, is_train=is_train, act=tf.nn.leaky_relu, name="d_2_bn_1")
		deconv3 = tl.layers.ConcatLayer([conv3, deconv3], concat_dim=3, name="concat_2")
		deconv3 = tl.layers.Conv2d(deconv3, n_filter=64, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv3")
		deconv3 = tl.layers.BatchNormLayer(deconv3, is_train=is_train, act=tf.nn.leaky_relu, name="d_2_bn_2")
		
		# 32*32*32
		deconv2 = tl.layers.DeConv2d(deconv3, n_filter=32, filter_size=(3, 3), strides=(2, 2), name="deconv2")
		deconv2 = tl.layers.BatchNormLayer(deconv2, is_train=is_train, act=tf.nn.leaky_relu, name="d_3_bn_1")
		deconv2 = tl.layers.ConcatLayer([conv2, deconv2], concat_dim=3, name="concat_3")
		deconv2 = tl.layers.Conv2d(deconv2, n_filter=32, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv2")
		deconv2 = tl.layers.BatchNormLayer(deconv2, is_train=is_train, act=tf.nn.leaky_relu, name="d_3_bn_2")
		
		# 64*64*16
		deconv1 = tl.layers.DeConv2d(deconv2, n_filter=16, filter_size=(3, 3), strides=(2, 2), name="deconv1")
		deconv1 = tl.layers.BatchNormLayer(deconv1, is_train=is_train, act=tf.nn.leaky_relu, name="d_4_bn_1")
		deconv1 = tl.layers.ConcatLayer([conv1, deconv1], concat_dim=3, name="concat_4")
		deconv1 = tl.layers.Conv2d(deconv1, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="conv_after_deconv1")
		net = tl.layers.BatchNormLayer(deconv1, is_train=is_train, act=tf.nn.relu, name="d_4_bn_2")

		for i in range(2):
			net = tl.layers.Conv2d(net, n_filter=256, filter_size=(3, 3), strides=(2, 2), name="sub_%d_c" % i)
			net = tl.layers.SubpixelConv2d(net, scale=2, act=tf.nn.relu, name="sub_%d_s" % i)
			# net = tl.layers.ElementwiseLayer([nn, net], combine_fn=tf.add, name="sub_%d_a" %i)
		
		
		# 64*64*3
		img_out = tl.layers.Conv2d(net, n_filter=3, filter_size=(3, 3), strides=(1, 1), act=tf.nn.tanh, name="img_out")
		return img_out
		



def network_d(image_input, reuse, is_train):
	with tf.variable_scope('network_d', reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)
		net = tl.layers.InputLayer(inputs=image_input, name="input_layer")
		net = tl.layers.Conv2d(net, n_filter=16, filter_size=(3, 3), strides=(2, 2), name="pre/conv")

		net = tl.layers.Conv2d(net, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="b1/c")
		net = tl.layers.BatchNormLayer(net, is_train=is_train, act=tf.nn.relu, name="b1/b")

		net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), name="pool1")

		net = tl.layers.Conv2d(net, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="b2/c")
		net = tl.layers.BatchNormLayer(net, is_train=is_train, act=tf.nn.relu, name="b2/b")

		net = tl.layers.Conv2d(net, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="b3/c")
		net = tl.layers.BatchNormLayer( net, is_train=is_train, act=tf.nn.relu, name="b3/b")

		net = tl.layers.MaxPool2d(net, filter_size=(3, 3), strides=(2, 2), name="pool2")

		net = tl.layers.Conv2d(net, n_filter=16, filter_size=(3, 3), strides=(1, 1), name="b4/c")
		net = tl.layers.BatchNormLayer(net, is_train=is_train, act=tf.nn.relu, name="b4/b")

		net = tl.layers.Conv2d(net, n_filter=16, filter_size=(3, 3), strides=(2, 2), name="b5/c")
		net = tl.layers.BatchNormLayer(net, is_train=is_train, act=tf.nn.relu, name="b5/b")

		net = tl.layers.FlattenLayer(net, name="flatten")

		net = tl.layers.DenseLayer(net, n_units=256, act=tf.nn.relu, name="dense512")
		pro = tl.layers.DenseLayer(net, n_units=1, act=tf.nn.relu, name="dense1/p")
		return net, pro


def Vgg19_simple_api(rgb, reuse):
	"""
	Build the VGG 19 Model

	Parameters
	-----------
	rgb : rgb image placeholder [batch, height, width, 3] values scaled [0, 1]
	"""
	VGG_MEAN = [103.939, 116.779, 123.68]
	with tf.variable_scope("VGG19", reuse=reuse) as vs:
		tl.layers.set_name_reuse(reuse)
		start_time = time.time()
		print("[*] VGG19 model build start...")

		rgb_scaled = rgb * 255.0
		# Convert RGB to BGR
		if tf.__version__ <= '0.11':
			red, green, blue = tf.split(3, 3, rgb_scaled)
		else:  # TF 1.0
			# print(rgb_scaled)
			red, green, blue = tf.split(rgb_scaled, 3, 3)
		assert red.get_shape().as_list()[1:] == [224, 224, 1]
		assert green.get_shape().as_list()[1:] == [224, 224, 1]
		assert blue.get_shape().as_list()[1:] == [224, 224, 1]
		if tf.__version__ <= '0.11':
			bgr = tf.concat(3, [blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2],])
		else:
			bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2],], axis=3)
		assert bgr.get_shape().as_list()[1:] == [224, 224, 3]
		""" input layer """
		net_in = tl.layers.InputLayer(bgr, name='input')
		""" conv1 """
		network = tl.layers.Conv2d(net_in, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv1_1')
		network = tl.layers.Conv2d(network, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu,  padding='SAME', name='conv1_2')
		network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool1')
		""" conv2 """
		network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_1')
		network = tl.layers.Conv2d(network, n_filter=128, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv2_2')
		network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool2')
		""" conv3 """
		network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_1')
		network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_2')
		network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_3')
		network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv3_4')
		network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool3')
		""" conv4 """
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_1')
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_2')
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_3')
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv4_4')
		network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool4')  # (batch_size, 14, 14, 512)
		conv = network
		""" conv5 """
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_1')
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_2')
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME',  name='conv5_3')
		network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3, 3), strides=(1, 1), act=tf.nn.relu, padding='SAME', name='conv5_4')
		network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2), padding='SAME', name='pool5')  # (batch_size, 7, 7, 512)
		""" fc 6~8 """
		network = tl.layers.FlattenLayer(network, name='flatten')
		network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc6')
		network = tl.layers.DenseLayer(network, n_units=4096, act=tf.nn.relu, name='fc7')
		network = tl.layers.DenseLayer(network, n_units=1000, act=tf.identity, name='fc8')

		print("[*] model build finished: %fs\n" % (time.time() - start_time))
		return network, conv
