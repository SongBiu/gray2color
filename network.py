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

		# 64*64*16
		conv1 = convWithBN(image_gray, 16, [3,1], is_train, "1_1")
		conv1 = convWithBN(conv1, 16, [3,1], is_train, "1_2")

		# 32*32*32
		conv2 = convWithBN(conv1, 32, [3,2], is_train, "2_1")
		conv2 = convWithBN(conv2, 32, [3,1], is_train, "2_2")
		
		# 16*16*64
		conv3 = convWithBN(conv2, 64, [3,2], is_train, "3_1")
		conv3 = convWithBN(conv3, 64, [3,1], is_train, "3_2")

		# 8*8*128
		conv4 = convWithBN(conv3, 128, [3,2], is_train, "4_1")
		conv4 = convWithBN(conv4, 128, [3,1], is_train, "4_2")
	   
		# 4*4*256
		conv5 = convWithBN(conv4, 256, [3,2], is_train, "5_1")
		conv5 = convWithBN(conv5, 256, [3,1], is_train, "5_2")
		
		'''
		architecture of deconv-layer:
			deconv(upsample) => 
			concat(with corresponding conv) => 
			conv(1*1 conv)
		'''
		# 8*8*128
		deconv4 = deconvWithBN(conv5, 128, [3,2], is_train, "4_1")
		deconv4 = tf.concat([deconv4,conv4], axis=3)
		deconv4 = convWithBN(deconv4, 128, [3,1], is_train, "de4_2")
		
		# 16*16*64
		deconv3 = deconvWithBN(deconv4, 64, [3,2], is_train, "3_1")
		deconv3 = tf.concat([deconv3,conv3], axis=3)
		deconv3 = convWithBN(deconv3, 64, [3,1], is_train, "de3_2")
		
		# 32*32*32
		deconv2 = deconvWithBN(deconv3, 32, [3,2], is_train, "2_1")
		deconv2 = tf.concat([deconv2,conv2], axis=3)
		deconv2 = convWithBN(deconv2, 32, [3,1], is_train, "de2_2")
		
		# 64*64*16
		deconv1 = deconvWithBN(deconv2, 16, [3,2], is_train, "1_1")
		deconv1 = tf.concat([deconv1,conv1], axis=3)
		deconv1 = convWithBN(deconv1, 16, [3,1], is_train, "de1_2")

		# 64*64*3
		img_out = tf.layers.conv2d_transpose(inputs=deconv1, filters=3, kernel_size=1, 
			strides=1, padding="same", activation=tf.nn.relu, name="out")
		img_out = tf.nn.tanh(img_out)
		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "network_g")
		
		return img_out, var_list
		

def network_d(image_input, is_train, keep_prob, reuse=None):

	with tf.variable_scope('network_d', reuse=reuse) as vs:

		# 64*64*16
		conv = convWithBN(image_input, 16, [3,1], is_train, "1_1")
		conv = convWithBN(conv, 16, [3,1], is_train, "1_2")
		conv = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, 
			padding="same", name="pool1")

		# 32*32*32
		conv = convWithBN(conv, 32, [3,1], is_train, "2_1")
		conv = convWithBN(conv, 32, [3,1], is_train, "2_2")
		conv = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, 
			padding="same", name="pool2")

		# 16*16*64
		conv = convWithBN(conv, 64, [3,1], is_train, "3_1")
		conv = convWithBN(conv, 64, [3,1], is_train, "3_2")
		conv = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, 
			padding="same", name="pool3")

		# 8*8*128
		conv = convWithBN(conv, 128, [3,1], is_train, "4_1")
		conv = convWithBN(conv, 128, [3,1], is_train, "4_2")
		conv = tf.layers.max_pooling2d(inputs=conv, pool_size=2, strides=2, 
			padding="same", name="pool4")

		# 4*4*128 => dense
		conv_flat = tf.reshape(conv, [-1,4*4*128], name="flat")
		conv = tf.layers.dense(inputs=conv, units=256, activation=tf.nn.relu, name="dense")

		# dropout
		if not is_train:
			keep_prob = 1
		conv = tf.nn.dropout(conv, keep_prob)
		conv = tf.layers.dense(inputs=conv, units=1, name="logits")

		var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "network_d")

		return conv, var_list