import os
import time
import random
import tensorflow as tf
import tensorlayer as tl
import function as func
import numpy as np
import network
import time
import argparse
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
image_size = 32
batch_size = 10
lr_init = 1.
n_epoch_init = 10
n_epoch = 3000
beta1 = 0.9
decay_round = 200
save_step = 10
checkpoint_path = ""
lr_decay = 0.1
total = 0
logging.basicConfig(filename="log", level=logging.INFO)

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-s", "--size", help="choose the size of image", type=int, default=64)
	parser.add_argument("-b", "--batchSize", help="input the batch size", type=int, default=30)
	parser.add_argument("-p", "--checkpointPath", help="input the path of checkpoint", type=str, default="ckp")
	parser.add_argument("-t", "--testStep", help="the step of test", type=int, default=10)
	parser.add_argument("-a", "--saveStep",  help="the step of save checkpoint", type=int,  default=10)
	parser.add_argument("-n", "--number", help="the number of train image", type=int, default=12000)
	args = parser.parse_args()
	return args

def train():
	global image_size, batch_size, lr_init, beta1, n_epoch_init, n_epoch, lr_decay, decay_round
	global save_step, checkpoint_path
	save_cnt = 0
	tl.files.exists_or_mkdir(checkpoint_path)
	
	image_gray = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 1], name="image_gray")
	image_color = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3],  name="image_color")
	

	"""GAN's train inference"""
	net_g = network.network_g(image_gray=image_gray, is_train=True, reuse=False)
	d_input_real = tf.concat([image_gray, image_color], axis=3)
	d_input_fake = tf.concat([image_gray, net_g.outputs*255], axis=3)
	net_d, logits_real = network.network_d(image_input=d_input_real, is_train=True, reuse=False)
	_, logits_fake = network.network_d(image_input=d_input_fake, is_train=True, reuse=True)


	"""VGG's inference"""
	fake_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0)
	real_224 = tf.image.resize_images(image_color, size=[224, 224], method=0)

	"""loss"""
	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	with tf.control_dependencies(update_ops):
		d_loss_1 = tl.cost.sigmoid_cross_entropy(logits_real.outputs, tf.ones_like(logits_real.outputs))
		d_loss_2 = tl.cost.sigmoid_cross_entropy(logits_fake.outputs, tf.zeros_like(logits_fake.outputs))
		D_loss = d_loss_1 + d_loss_2
		g_gan_loss = tl.cost.sigmoid_cross_entropy(logits_fake.outputs, tf.ones_like(logits_fake.outputs))
		g_mse_loss = tf.reduce_mean(tf.losses.mean_squared_error(image_color, net_g.outputs*255))
		G_loss = g_gan_loss + g_mse_loss

		"""train op"""
		G_var = tl.layers.get_variables_with_name("network_g", train_only=True, printable=False)
		D_var = tl.layers.get_variables_with_name("network_d", train_only=True, printable=False)
		with tf.variable_scope('learn_rate'):
			lr_v = tf.Variable(lr_init, trainable=False)
		G_init_optimizer = tf.train.AdadeltaOptimizer(lr_v).minimize(g_mse_loss, var_list=G_var)
		D_optimizer = tf.train.AdadeltaOptimizer(lr_v).minimize(D_loss, var_list=D_var)
		G_optimizer = tf.train.AdadeltaOptimizer(lr_v).minimize(G_loss, var_list=G_var)

	"""train"""
	with tf.Session() as sess:
		tl.layers.initialize_global_variables(sess)

		for epoch in range(n_epoch_init):
			img_list = func.init_list(image_size)
			epoch_time = time.time()
			n_iter, total_g_loss = 0, 0
			for idx in range(0, total, batch_size):
				step_time = time.time()
				if idx + batch_size > total:
					break
				input_gray, input_color = func.load(size=image_size, start=idx, number=batch_size, img_list=img_list)
				errG, _ = sess.run([g_mse_loss, G_init_optimizer], feed_dict={image_gray: input_gray, image_color: input_color})
				print "[TF] Epoch [%2d/%2d] %4d  time: %4.4fs, g_loss: %.8f" % (epoch, n_epoch_init, n_iter, time.time() - step_time, errG)
				total_g_loss += errG
				n_iter += 1
			log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_g_loss / n_iter)
			logging.info(log)
			
		for epoch in range(n_epoch - n_epoch_init):
			if epoch != 0 and (epoch % decay_round == 0):
				new_lr_decay = lr_decay ** (epoch // decay_round)
				sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
				log = "[*] Epoch:[%2d/%2d] new learning rate: %f (for GAN)" % (epoch, n_epoch - n_epoch_init, lr_init * new_lr_decay)
				logging.info(log)
			elif epoch == 0:
				sess.run(tf.assign(lr_v, lr_init))
				log = "[%s] init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (time.ctime(), lr_init, decay_round, lr_decay)
				logging.info(log)
				
			img_list = func.init_list(image_size)
			n_iter, total_d_loss, total_g_loss, total_fake_loss = 0, 0, 0, 0
			epoch_time = time.time()
			for idx in range(0, total, batch_size):
				step_time = time.time()
				if idx + batch_size > total:
					break
				input_gray, input_color = func.load(size=image_size, start=idx, number=batch_size, img_list=img_list)
				errG, errD, _, _, _, _, d_fake_loss = sess.run([G_loss, D_loss, G_optimizer, G_optimizer, G_optimizer, D_optimizer, d_loss_2], feed_dict={image_gray: input_gray, image_color: input_color})
				print "[TF] Epoch [%2d/%2d] %4d  time: %4.4fs, d_loss: %.8f(fake_loss: %.8f) g_loss: %.8f" % (epoch, n_epoch - n_epoch_init, n_iter, time.time() - step_time, errD, d_fake_loss, errG)
				total_d_loss += errD
				total_g_loss += errG
				total_fake_loss += d_fake_loss
				n_iter += 1
			log = "[%s] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f(fake_loss: %.8f) g_loss: %.8f" % (time.ctime(), epoch, n_epoch - n_epoch_init, time.time() - epoch_time, total_d_loss / n_iter, total_fake_loss / n_iter, total_g_loss / n_iter)
			logging.info(log)

			if epoch != 0 and (epoch + 1) % save_step == 0:
				log =  "[%s] epoch %d, save as %s/g_%d.npz" % (time.ctime(), epoch, checkpoint_path, save_cnt%10)
				logging.info(log)
				tl.files.save_npz(net_g.all_params, name="%s/g_%d.npz" % (checkpoint_path, save_cnt % 10), sess=sess)
				tl.files.save_npz(net_d.all_params, name="%s/d_%d.npz" % (checkpoint_path, save_cnt % 10), sess=sess)
				save_cnt += 1
			else:
				print "[*] sorry.path=%s" % checkpoint_path


def main():
	global image_size, batch_size, total, save_step, checkpoint_path
	args = get_args()
	image_size = args.size
	batch_size = args.batchSize
	total = args.number
	checkpoint_path = args.checkpointPath
	save_step = args.saveStep
	train()


if __name__ == "__main__":
	main()
