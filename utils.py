"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import matplotlib
import matplotlib.pyplot
import numpy as np
from time import gmtime, strftime
from six.moves import xrange
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

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

def save_images2(image, image_path):
    image = np.reshape(image,[96,96])
    return imsave2(inverse_transform(image),image_path)

def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def imsave2(image,path):
    return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width,
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

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
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

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
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config, option):
  image_frame_dim = int(math.ceil(config.batch_size ** .5))
  if option == 0:
    for i in range(600):
      z_sample = np.random.uniform(-1, 1, size=(config.batch_size, dcgan.z_dim))
      samples = sess.run(dcgan.sampler2, feed_dict={dcgan.z: z_sample})

      for idx, im in enumerate(samples):
        save_images2(im, './samples/test/test_%s.png' % ('' + str(i) + '_' + str(idx)))
  elif option == 1:
    values = np.arange(0, 1, 1./config.batch_size)
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
  elif option == 2:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in [random.randint(0, 99) for _ in xrange(100)]:
      print(" [*] %d" % idx)
      z = np.random.uniform(-0.2, 0.2, size=(dcgan.z_dim))
      z_sample = np.tile(z, (config.batch_size, 1))
      #z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      if config.dataset == "mnist":
        y = np.random.choice(10, config.batch_size)
        y_one_hot = np.zeros((config.batch_size, 10))
        y_one_hot[np.arange(config.batch_size), y] = 1

        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample, dcgan.y: y_one_hot})
      else:
        samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})

      try:
        make_gif(samples, './samples/test_gif_%s.gif' % (idx))
      except:
        save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
  elif option == 3:
    values = np.arange(0, 1, 1./config.batch_size)
    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample):
        z[idx] = values[kdx]

      samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample})
      make_gif(samples, './samples/test_gif_%s.gif' % (idx))
  elif option == 4:
    image_set = []
    values = np.arange(0, 1, 1./config.batch_size)

    for idx in xrange(100):
      print(" [*] %d" % idx)
      z_sample = np.zeros([config.batch_size, dcgan.z_dim])
      for kdx, z in enumerate(z_sample): z[idx] = values[kdx]

      image_set.append(sess.run(dcgan.sampler, feed_dict={dcgan.z: z_sample}))
      make_gif(image_set[-1], './samples/test_gif_%s.gif' % (idx))

    new_image_set = [merge(np.array([images[idx] for images in image_set]), [10, 10]) \
        for idx in range(64) + range(63, -1, -1)]
    make_gif(new_image_set, './samples/test_gif_merged.gif', duration=8)


def image_manifold_size(num_images):
  manifold_h = int(np.floor(np.sqrt(num_images)))
  manifold_w = int(np.ceil(np.sqrt(num_images)))
  assert manifold_h * manifold_w == num_images
  return manifold_h, manifold_w


# Generate  latent that generates similar query image ... optimize using Residual loss (Tee)
# Detect if being a normal or anomalous image.
def generate_latent_for_query(sess, dcgan, query_image, inputClass, FLAGS, OPTION, batch_size=1):
  global global_counter
  global_counter = 0
  try:
    os.makedirs(os.path.join('./tests', 'MIAS'))
  except:
    print('Dir exsists')
    dcgan.batch_size = batch_size

    # define a tensorflow varb
    w = tf.Variable(initial_value=tf.random_uniform(minval=-1, maxval=1, shape=[batch_size, dcgan.z_dim]),
                    name='qnoise')

    samples = dcgan.sampler(w)
    samples = (samples+1)/2
    # get activations from discriminator for L loss
    # 	query = tf.convert_to_tensor(query_im, dtype=tf.float32)
    query = tf.placeholder(shape=[1, 64, 64, 1], dtype=tf.float32)
    # print(query.get_shape, query_im.shape)
    # print("samples",samples.get_shape)
    _, _, real = dcgan.discriminator(query, reuse=True)
    _, _, fake = dcgan.discriminator(samples, reuse=True)

    # define loss funtion and optimizer
    Resloss = tf.reduce_mean(tf.abs(samples - query))
    DiscLoss = tf.reduce_mean(tf.abs(real - fake))
    loss = 2*Resloss + 0.5*DiscLoss
    # Training params
    learning_rate = 0.0007
    beta1 = 0.7
    optim = tf.train.AdamOptimizer(learning_rate, beta1=beta1) \
      .minimize(loss, var_list=[w])

    # init vars
    w_init = w.initializer
    sess.run(w_init)
    # tf.variables_initializer([w]).run(session = sess)
    print(sess.run(tf.report_uninitialized_variables()))
    Adam_initializers = [var.initializer for var in tf.global_variables() if 'qnoise/Adam' in var.name]
    sess.run(Adam_initializers)
    beta_initializers = [var.initializer for var in tf.global_variables() if 'beta1_power' in var.name]
    sess.run(beta_initializers)
    beta_initializers = [var.initializer for var in tf.global_variables() if 'beta2_power' in var.name]
    sess.run(beta_initializers)

    x = 1
    avg = 0
    arr = os.listdir(os.getcwd() + "/data/normal_mias2/")
    #while (x != 0):
    for im in arr:
      if (inputClass is not None):
        q, _ = inputClass.next_batch(1)
        print("using next batch")
      else:
        print(im)
        q = scipy.misc.imread(os.getcwd() + "/data/normal_mias2/" + im, flatten=True)
        q = np.asarray(q)
        q /= 255
      # q = transform(q, 64, 64)
      # q = np.expand_dims(q,axis=2)
      # q = np.expand_dims(q, axis=0)

      q = np.asarray(q)
      q = np.expand_dims(q,axis=0)
      q = np.expand_dims(q,axis=3)
      print(q.shape)
      loss, R, D = query_noise2(dcgan, sess, q, w, optim, loss, query, Resloss, DiscLoss)
      # x = input("press 0 to quit")
      global_counter += 1

      avg += loss
      print(loss, R, D)
      print('AVerage Loss', avg / global_counter)

      # define loss funtion and optimizer
      Resloss = tf.reduce_mean(tf.abs(samples - query))
      DiscLoss = tf.reduce_mean(tf.abs(real - fake))
      loss = 2*Resloss + 0.5*DiscLoss
      w_init = w.initializer
      sess.run(w_init)
    return False




# Takes a query image, returns the image re-created by the GAN
def query_noise2(dcgan, sess, query_im, w, optim, loss, query, Res_loss, Disc_loss, batch_size=1, noise_size=100):
  matplotlib.pyplot.ioff()

  losses = []
  # backprop over noise to minimize loss(r iterations)
  r = 2000
  for i in range(r):
    _, current_loss, noise = sess.run([optim, loss, w], feed_dict={query: query_im})
    losses.append(current_loss)
  R, D = sess.run([Res_loss, Disc_loss], {query: query_im})
  print(current_loss)

  matplotlib.pyplot.figure(1)
  matplotlib.pyplot.plot(np.arange(0, r, 1), losses)
  if (global_counter % 2 == 0):
    matplotlib.pyplot.savefig(str(global_counter) + '.png')

  # matplotlib.pyplot.show(block = True)

  #z_sample = np.random.uniform(-1, 1, size=(64 , dcgan.z_dim))

  z_sample = noise
  samples = dcgan.sampler(w)
  samples = sess.run(samples, feed_dict={w: z_sample})
  # save_images(samples, [64, 64],
  # './tests/' + 'MIAS' + '/Query-recreation'+str(global_counter))
  # save_images(query_im, [64, 64],
  # './tests/' + 'MIAS' + '/Query'+str(global_counter))

  samples = np.asarray(samples[0])
  samples2 = np.reshape(samples,[64,64])
  #samples = inverse_transform(samples2)

  query_im = query_im.squeeze()
  samples = samples.squeeze()
  samples = (samples+1)/2
  res_im = np.absolute((samples - query_im))

  print(samples)
  print('.....')
  print(query_im)

  tmp = np.concatenate([query_im, samples, res_im], axis=1)

  # samples = inverse_transform(samples)
  print(samples.max(), samples.min())
  print(query_im.max(), query_im.min())
  print(res_im.max(), res_im.min())
  print(tmp.shape)
  maxi = np.max(tmp)
  mini = np.min(tmp)

  # matplotlib.image.imsave('./tests/' + 'MIAS' + '/both'+str(global_counter)+'.png', np.reshape(tmp, (tmp.shape[0], tmp.shape[1])), cmap="gray",vmin=0,vmax=1)
  matplotlib.image.imsave(
    './tests/' + 'MIAS' + '/both' + str(global_counter) + 'normalized-' + str(current_loss) + '-.png',
    np.reshape(tmp, (tmp.shape[0], tmp.shape[1])), cmap="gray")
  print(query_im.shape)

  return current_loss, R, D


def get_lables(batch_images):
  labels = []
  for image in batch_images:
    image = image.astype('float32')
    if image.max() > 1.0:
      image /= 255.0
    label = []
    label.append(np.mean(image))
    label.append(np.std(image))
    labels.append(label)
  return labels
