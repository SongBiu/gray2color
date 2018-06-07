import os
import time
import random
import tensorflow as tf
import function as func
import numpy as np
import network
import time
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


image_size = 32
batch_size = 10
lr_init = 3e-3
n_epoch_init = 10
n_epoch = 300
beta1 = 0.9
decay_round = 200
save_step = 10
checkpoint_path = ""
lr_decay = 0.1
total = 0


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


def exists_or_mkdir(dirname):
    
    if not os.path.exists(dirname):
        os.mkdir(dirname)


def train():

    global image_size, batch_size, lr_init, beta1, n_epoch_init, n_epoch, lr_decay, decay_round
    global save_step, checkpoint_path
    save_cnt = 0
    exists_or_mkdir(checkpoint_path)
    
    image_gray = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 1], name="image_gray")
    image_color = tf.placeholder(dtype=tf.float32, shape=[batch_size, image_size, image_size, 3],  name="image_color")
    
    net_g, G_var = network.network_g(image_gray=image_gray, is_train=True)
    
    d_input_real = tf.concat([image_gray, image_color], axis=3)
    d_input_fake = tf.concat([image_gray, net_g*255], axis=3)
    logits_real, D_var = network.network_d(image_input=d_input_real, is_train=True, keep_prob=0.5)
    logits_fake, _ = network.network_d(image_input=d_input_fake, is_train=True, keep_prob=0.5, reuse=True)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits_real), logits=logits_real))
        d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(logits_fake), logits=logits_fake))
        D_loss = d_loss_real + d_loss_fake
        
        g_loss_gan = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(logits_fake), logits=logits_fake))
        g_loss_mse = tf.losses.mean_squared_error(image_color, net_g*255)
        G_loss = g_loss_gan + 3e-4*g_loss_mse

        G_init_optimizer = tf.train.AdadeltaOptimizer(0.1).minimize(g_loss_mse, var_list=G_var)
        D_optimizer = tf.train.AdadeltaOptimizer(lr_init).minimize(D_loss, var_list=D_var)
        G_optimizer = tf.train.AdadeltaOptimizer(lr_init).minimize(G_loss, var_list=G_var)

    saver = tf.train.Saver(var_list=G_var, max_to_keep=10)
    """train"""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # pre-train for G
        for epoch in range(n_epoch_init):
            # shuffle
            img_list = func.init_list(image_size)
            epoch_time = time.time()
            n_iter, total_g_loss = 0, 0
            for idx in range(0, total, batch_size):
                step_time = time.time()
                if idx + batch_size > total:
                    break
                input_gray, input_color = func.load(size=image_size, start=idx, number=batch_size, img_list=img_list)
                errG, _ = sess.run([g_loss_mse, G_init_optimizer], feed_dict={image_gray: input_gray, image_color: input_color})
                print("[TF] Epoch [%2d/%2d] %4d  time: %4.4fs, g_loss: %.8f" % (epoch, n_epoch_init, n_iter, time.time() - step_time, errG))
                total_g_loss += errG
                n_iter += 1
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f" % (epoch, n_epoch_init, time.time() - epoch_time, total_g_loss / n_iter)
            print(log)

        for epoch in range(n_epoch_init, n_epoch):
            # shuffle
            img_list = func.init_list(image_size)
            n_iter, total_d_loss, total_g_loss = 0, 0, 0
            epoch_time = time.time()
            for idx in range(0, total, batch_size):
                step_time = time.time()
                if idx + batch_size > total:
                    break
                input_gray, input_color = func.load(size=image_size, start=idx, number=batch_size, img_list=img_list)
                errD, _ = sess.run([D_loss, D_optimizer], feed_dict={image_gray: input_gray, image_color: input_color})
                errG, _ = sess.run([G_loss, G_optimizer], feed_dict={image_gray: input_gray, image_color: input_color})
                errG, _ = sess.run([G_loss, G_optimizer], feed_dict={image_gray: input_gray, image_color: input_color})
                errG, _ = sess.run([G_loss, G_optimizer], feed_dict={image_gray: input_gray, image_color: input_color})       
                print("[TF] Epoch [%2d/%2d] %4d  time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, n_iter, time.time() - step_time, errD, errG))
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_d_loss / n_iter, total_g_loss / n_iter)
            print(log)

            if epoch != 0 and (epoch + 1) % save_step == 0:
                print("[*] save ! path=%s" % checkpoint_path)
                saver.save(sess, "./"+checkpoint_path, global_step=save_cnt)
                save_cnt += 1


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