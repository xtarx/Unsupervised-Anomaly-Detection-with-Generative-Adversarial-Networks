import os
import numpy as np
import inception_score

from model import DCGAN
from utils import *

import tensorflow as tf

# suppress warning s
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.logging.set_verbosity(tf.logging.ERROR)
flags = tf.app.flags
flags.DEFINE_integer("epoch", 3, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None,
                     "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None,
                     "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("crop", True, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS


def main(_):
    # pp.pprint(flags.FLAGS.__flags)

    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True

    with tf.Session(config=run_config) as sess:

        dcgan = DCGAN(
            sess,
            input_width=FLAGS.input_width,
            input_height=FLAGS.input_height,
            output_width=FLAGS.output_width,
            output_height=FLAGS.output_height,
            batch_size=FLAGS.batch_size,
            sample_num=FLAGS.batch_size,
            dataset_name=FLAGS.dataset,
            input_fname_pattern=FLAGS.input_fname_pattern,
            crop=FLAGS.crop,
            checkpoint_dir=FLAGS.checkpoint_dir,
            sample_dir=FLAGS.sample_dir)

        # show_all_variables()

        if FLAGS.train:
            dcgan.train(FLAGS)
        else:
            if not dcgan.load(FLAGS.checkpoint_dir)[0]:
                raise Exception("[!] Train a model first, then run test mode")

        # Below is codes for visualization
        OPTION = 0
        # visualize(sess, dcgan, FLAGS, OPTION)

        # gen_list = generate_imgs(sess, dcgan, FLAGS)
        #
        # print(len(gen_list))
        # print(gen_list[0].shape)
        # for i in range(21):
        #     scipy.misc.imsave('./samples/testGANs/'+str(i)+'.png', gen_list[i])
        #


        test_list = []
        gen_list = []

        arr = os.listdir(os.getcwd() + "/data/test/")
        for test_image in arr:
            query_image_path = os.getcwd() + "/data/test/" + test_image
            c_dim = imread(query_image_path).shape[-1]
            grayscale = (c_dim == 1)
            q_img = get_image(query_image_path,
                              input_height=FLAGS.input_height,
                              input_width=FLAGS.input_width,
                              resize_height=FLAGS.output_height,
                              resize_width=FLAGS.output_width,
                              crop=FLAGS.crop,
                              grayscale=grayscale);
            # q_img *= 255.0
            test_list.append(q_img);

        arr = os.listdir(os.getcwd() + "/data/testGANs/")
        arr=arr[1:]
        print(arr)
        for test_image in arr:
            query_image_path = os.getcwd() + "/data/testGANs/" + test_image
            c_dim = imread(query_image_path).shape[-1]
            grayscale = (c_dim == 1)
            q_img = get_image(query_image_path,
                              input_height=64,
                              input_width=64,
                              resize_height=FLAGS.output_height,
                              resize_width=FLAGS.output_width,
                              crop=False,
                              grayscale=grayscale);
            # q_img *= 255.0
            gen_list.append(q_img);

    # print((test_list[0]))

    real = inception_score.get_inception_score(test_list)
    print('real inception: {}'.format(real))
    print("------------")
    fake = inception_score.get_inception_score(gen_list)
    print('fake inception: {}'.format(fake))



    # print(query_image);
    # generate_latent_for_query(sess, dcgan, query_image, FLAGS=FLAGS, OPTION=OPTION)


    # calculate inception score


if __name__ == '__main__':
    tf.app.run()
