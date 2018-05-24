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

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
image_size = 32
batch_size = 10
lr_init = 1e-1
n_epoch_init = 10
n_epoch = 100
beta1 = 0.9
decay_round = 200
save_step = 10
checkpoint_path = ""
lr_decay = 0.1
total = 0
mode = ""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        help="choose train or eval",
        type=str,
        default="train",
        choices=["train", "eval"])
    parser.add_argument(
        "-s", "--size", help="choose the size of image", type=int, default=224)
    parser.add_argument(
        "-b", "--batchSize", help="input the batch size", type=int, default=10)
    parser.add_argument(
        "-p",
        "--checkpointPath",
        help="input the path of checkpoint",
        type=str,
        default="ckp")
    parser.add_argument(
        "-t", "--testStep", help="the step of test", type=int, default=10)
    parser.add_argument(
        "-a",
        "--saveStep",
        help="the step of save checkpoint",
        type=int,
        default=500)
    parser.add_argument(
        "-n",
        "--number",
        help="the number of train image",
        type=int,
        default=1882)
    args = parser.parse_args()
    return args


def train():
    global image_size, batch_size, lr_init, beta1, n_epoch_init, n_epoch, lr_decay, decay_round
    global save_step, checkpoint_path, mode
    image_gray = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, image_size, image_size, 1],
        name="image_gray")
    image_color = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, image_size, image_size, 3],
        name="image_color")
    """GAN's train inference"""
    net_g = network.network_g(
        image_gray=image_gray, is_train=True, reuse=False)
    net_d, logitsReal = network.network_d(
        image_input=image_color, is_train=True, reuse=False)
    _, logitsFake = network.network_d(
        image_input=net_g.outputs, is_train=True, reuse=True)
    """GAN's test inference"""
    net_g_test = network.network_g(
        image_gray=image_gray, is_train=False, reuse=True)
    """VGG's inference"""
    net_vgg, vgg_real_img = network.Vgg19_simple_api(image_color, reuse=False)
    _, vgg_fake_img = network.Vgg19_simple_api(net_g.outputs, reuse=True)
    """loss"""
    # d_loss1 = tl.cost.sigmoid_cross_entropy(logitsReal, tf.ones_like())
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        D_loss = tl.cost.cross_entropy(
            logitsFake.outputs) + tl.cose.cross_entropy(1 - logitsReal.outputs)
        g_gan_loss = tl.cost.cross_entropy(1 - logitsFake.outputs)
        g_vgg_loss = tl.cost.mean_squared_error(
            vgg_fake_img.outputs, vgg_real_img.outputs, is_mean=True) / 1000
        g_mse_loss = tl.cost.mean_squared_error(
            net_g.outputs, image_color, is_mean=True) / (3 * 224**2)
        G_loss = g_mse_loss + g_gan_loss + g_vgg_loss
        """train op"""
        G_var = tl.layers.get_variables_with_name(
            "network_g", train_only=True, printable=False)
        D_var = tl.layers.get_variables_with_name(
            "network_d", train_only=False, printable=False)
        with tf.variable_scope('learn_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        G_init_optimizer = tf.train.GradientDescentOptimizer(1e-2).minimize(
            g_mse_loss, var_list=G_var)
        D_optimizer = tf.train.AdamOptimizer(
            lr_v, beta1=beta1).minimize(
                D_loss, var_list=D_var)
        G_optimizer = tf.train.AdamOptimizer(
            lr_v, beta1=beta1).minimize(
                G_loss, var_list=G_var)
    """train"""
    with tf.Session() as sess:
        tl.layers.initialize_global_variables(sess)
        npz = np.load("vgg19.npy", encoding='latin1').item()
        params = []
        for val in sorted(npz.items()):
            W = np.asarray(val[1][0])
            b = np.asarray(val[1][1])
            print("  Loading %s: %s, %s" % (val[0], W.shape, b.shape))
            params.extend([W, b])
        tl.files.assign_params(sess, params, net_vgg)
        print "[TF]	Global Variables initialized!"

        for epoch in range(n_epoch_init):
            epoch_time = time.time()
            n_iter, total_g_loss = 0, 0
            for idx in range(0, total, batch_size):
                step_time = time.time()
                if idx + batch_size > total:
                    break
                input_gray, input_color = func.load(
                    size=image_size, start=idx, number=batch_size)
                errG, _ = sess.run(
                    [G_loss, G_init_optimizer],
                    feed_dict={
                        image_gray: input_gray,
                        image_color: input_color
                    })
                print "[TF] Epoch [%2d/%2d] %4d  time: %4.4fs, g_loss: %.8f" % (
                    epoch, n_epoch_init, n_iter, time.time() - step_time, errG)
                total_g_loss += errG
                n_iter += 1
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f" % (
                epoch, n_epoch_init, time.time() - epoch_time,
                total_g_loss / n_iter)
            print log

        for epoch in range(n_epoch_init, n_epoch):
            n_iter, total_d_loss, total_g_loss = 0, 0, 0
            epoch_time = time.time()
            if epoch != 0 and epoch % decay_round == 0:
                new_lr_decay = lr_decay**(epoch // decay_round)
                sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
                print "[TF] new learning rate: %f (for GAN)" % (
                    lr_init * new_lr_decay)
            elif epoch == 0:
                sess.run(tf.assign(lr_v, lr_init))
                print "[TF] init learning rate: %f, decay_every_round: %d, lr_decay: %f (for GAN)" % (
                    lr_init, decay_round, lr_decay)
            for idx in range(0, total, batch_size):
                step_time = time.time()
                if idx + batch_size > total:
                    break
                input_gray, input_color = func.load(
                    size=image_size, start=idx, number=batch_size)
                print(
                    sess.run(
                        [logitsFake.outputs, 1 - logitsFake.outputs],
                        feed_dict={
                            image_gray: input_gray,
                            image_color: input_color
                        }))
                errD, _ = sess.run(
                    [D_loss, D_optimizer],
                    feed_dict={
                        image_gray: input_gray,
                        image_color: input_color
                    })
                errG, _ = sess.run(
                    [G_loss, G_optimizer],
                    feed_dict={
                        image_gray: input_gray,
                        image_color: input_color
                    })
                print "[TF] Epoch [%2d/%2d] %4d  time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
                    epoch, n_epoch, n_iter, time.time() - step_time, errD,
                    errG)
                total_d_loss += errD
                total_g_loss += errG
                n_iter += 1
            log = "[*] Epoch: [%2d/%2d] time: %4.4fs, d_loss: %.8f g_loss: %.8f" % (
                epoch, n_epoch, time.time() - epoch_time,
                total_d_loss / n_iter, total_g_loss / n_iter)
            print log

            if epoch != 0 and epoch % save_step == 0:
                tl.files.save_npz(
                    net_g.all_params,
                    name="%s/g_%d.npz" % (checkpoint_path, mode),
                    sess=sess)
                tl.files.save_npz(
                    net_d.all_params,
                    name="%s/d_%d.npz" % (checkpoint_path, mode),
                    sess=sess)


def main():
    global image_size, batch_size, total, save_step, checkpoint_path
    args = get_args()
    image_size = args.size
    batch_size = args.batchSize
    total = args.number
    checkpoint_path = args.checkpointPath
    save_step = args.saveStep
    mode = args.mode
    train()


if __name__ == "__main__":
    main()
