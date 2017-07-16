"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint

import matplotlib
import scipy.misc
import numpy as np
from time import gmtime, strftime
from six.moves import xrange

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1 / math.sqrt(k_w * k_h * x.get_shape()[-1])


def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
    image = imread(image_path, grayscale)
    return transform(image, input_height, input_width,
                     resize_height, resize_width, crop)


def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)


def imread(path, grayscale=False):
    if (grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3, 4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3] == 1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:, :, 0]
        return img
    else:
        raise ValueError('in merge(images,size) images parameter '
                         'must have dimensions: HxW or HxWx3 or HxWx4')


def imsave(images, size, path):
    image = np.squeeze(merge(images, size))
    return scipy.misc.imsave(path, image)


def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h) / 2.))
    i = int(round((w - crop_w) / 2.))
    return scipy.misc.imresize(
        x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
    if crop:
        cropped_image = center_crop(
            image, input_height, input_width,
            resize_height, resize_width)
    else:
        cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
    # return np.array(cropped_image) / 127.5 - 1.
    return np.array(cropped_image)


def inverse_transform(images):
    return (images + 1.) / 2.


def to_json(output_path, *layers):
    with open(output_path, "w") as layer_f:
        lines = ""
        for w, b, bn in layers:
            layer_idx = w.name.split('/')[0].split('h')[1]

            B = b.eval()

            if "lin/" in w.name:
                W = w.eval()
                depth = W.shape[1]
            else:
                W = np.rollaxis(w.eval(), 2, 0)
                depth = W.shape[0]

            biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
            if bn != None:
                gamma = bn.gamma.eval()
                beta = bn.beta.eval()

                gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
                beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
            else:
                gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
                beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

            if "lin/" in w.name:
                fs = []
                for w in W.T:
                    fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

                lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
            else:
                fs = []
                for w_ in W:
                    fs.append(
                        {"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

                lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2 ** (int(layer_idx) + 2), 2 ** (int(layer_idx) + 2),
                   W.shape[0], W.shape[3], biases, gamma, beta, fs)
        layer_f.write(" ".join(lines.replace("'", "").split()))


def generate_imgs(sess, dcgan, config):
    z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
    return (sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))


def visualize(sess, dcgan, config, option):
    image_frame_dim = int(math.ceil(config.batch_size ** .5))
    if option == 0:
        for i in range(10):
            z_sample = np.random.uniform(-0.5, 0.5, size=(config.batch_size, dcgan.z_dim))
            samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
            save_images(samples, [image_frame_dim, image_frame_dim],
                        './samples/test/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
    elif option == 1:
        values = np.arange(0, 1, 1. / config.batch_size)
        for idx in xrange(100):
            print(" [*] %d" % idx)
            z_sample = np.zeros([config.batch_size, dcgan.z_dim])
            for kdx, z in enumerate(z_sample):
                z[idx] = values[kdx]

            if config.dataset == "mnist":
                y = np.random.choice(10, config.batch_size)
                y_one_hot = np.zeros((config.batch_size, 10))
                y_one_hot[np.arange(config.batch_size), y] = 1

                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
            else:
                samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

            save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_arange_%s.png' % (idx))


def image_manifold_size(num_images):
    manifold_h = int(np.floor(np.sqrt(num_images)))
    manifold_w = int(np.ceil(np.sqrt(num_images)))
    assert manifold_h * manifold_w == num_images
    return manifold_h, manifold_w


# Generate  latent that generates similar query image ... optimize using Residual loss (Tee)
# Detect if being a normal or anomalous image.
# def generate_latent_for_query(sess, dcgan, query_image, inputClass, FLAGS, OPTION):
def generate_latent_for_query(sess, dcgan, query_image, FLAGS, OPTION):
    x = 1
    while (x):
        # q = scipy.misc.imread(query_image, flatten=True)
        # q = np.asarray(q)
        q = query_image
        print(q.shape)
        out = query_noise2(dcgan, sess, q)
        x = input("press 0 to quit")
    return False


# Takes a query image, returns the image re-created by the GAN
def query_noise2(dcgan, sess, query_im, batch_size=1, noise_size=100):
    dcgan.batch_size = batch_size
    # define a tensorflow var
    w = tf.Variable(initial_value=tf.random_normal(mean=0, stddev=2, shape=[batch_size, dcgan.z_dim]),
                    name='qnoise')
    samples = dcgan.sampler2(w)
    # get activations from discriminator for L loss
    # 	query = tf.convert_to_tensor(query_im, dtype=tf.float32)
    query = tf.placeholder(shape=[1, 64, 64, 1], dtype=tf.float32)
    # print(query.get_shape, query_im.shape)
    # print("samples",samples.get_shape)
    # _, _, real = dcgan.discriminator2(query, reuse=True)
    # _, _, fake = dcgan.discriminator2(samples, reuse=True)
    # define loss funtion and optimizer
    Resloss = tf.reduce_mean(tf.abs(samples - query))
    # DiscLoss = tf.reduce_mean(tf.abs(real - fake))
    DiscLoss = 0
    loss = Resloss + DiscLoss
    # Training params
    learning_rate = 0.0006
    beta1 = 0.7
    optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
        .minimize(loss, var_list=[w])
    # init vars
    tf.variables_initializer([w]).run()
    print(sess.run(tf.report_uninitialized_variables()))
    Adam_initializers = [var.initializer for var in tf.global_variables() if 'qnoise/Adam' in var.name]
    sess.run(Adam_initializers)
    beta_initializers = [var.initializer for var in tf.global_variables() if 'beta1_power' in var.name]
    sess.run(beta_initializers)
    beta_initializers = [var.initializer for var in tf.global_variables() if 'beta2_power' in var.name]
    sess.run(beta_initializers)
    losses = []
    # backprop over noise to minimize loss(500 iterations)
    for i in range(100):
        _, current_loss, noise = sess.run([optim, loss, w], feed_dict={query: query_im})
        losses.append(current_loss)
        print(current_loss)
    matplotlib.pyplot.plot(np.arange(0, 6000, 1), losses)
    matplotlib.pyplot.show(block=True)
    # 	for i in range(10):
    # 			z_sample = np.random.normal(0, 2, size=(1, dcgan.z_dim))
    # 			samples = dcgan.sampler2(w)
    # 			samples = sess.run(samples, feed_dict={w: z_sample})
    # 			save_images(samples, [64, 64],
    # 						 './tests/' + 'MIAS' + '/test_arange_%s.png' % (i))
    z_sample = noise
    samples = dcgan.sampler2(w)
    samples = sess.run(samples, feed_dict={w: z_sample})
    save_images(samples, [64, 64],
                './tests/' + 'MIAS' + '/Query-recreation.png')
    save_images(query_im, [64, 64],
                './tests/' + 'MIAS' + '/Query.png')
    cont = tf.concat([query, samples], axis=2)
    print(cont.get_shape)
    tmp = np.ones([1, 64, 128, 1])
    tmp = np.asarray(cont[0, :, :, 0])
    tmp = tmp.squeeze()
    print("shapezzz", tmp.shape)
    matplotlib.image.imsave('./tests/' + 'MIAS' + '/both.png', np.reshape(tmp, (tmp.shape[0], tmp.shape[1])),
                            cmap="gray")
    return current_loss
