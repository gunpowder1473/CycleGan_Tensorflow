# coding=utf-8
import tensorflow as tf
from GANNetwork.cycleGAN import CycleGAN
from common.common import getImg, torch_decay, getFiles, saveImg, imgPool, encode
import numpy as np
import random
import threading, os, time, cv2
from tensorflow.python import pywrap_tensorflow

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('input',
                           "horse2zebra/trainA/n02381460_175.jpg",
                           'The directory that input pictures are saved')
tf.app.flags.DEFINE_string('output', "out/14.jpg", 'The directory that validate pictures are saved')
tf.app.flags.DEFINE_string('checkpoint', "model/model_zebra.ckpt",
                           'The directory that trained network will be saved')
tf.app.flags.DEFINE_string('status',
                           "X2Y",
                           'The directory that input pictures are saved')
tf.app.flags.DEFINE_string('Norm', 'BATCH', 'Choose to use Batchnorm or instanceNorm')
tf.app.flags.DEFINE_bool('USE_E', False, 'Choose to use Edge or not')
tf.app.flags.DEFINE_integer('img_size', 256, 'The size of input img')
tf.app.flags.DEFINE_integer('ngf', 64, 'The number of gen filters in first conv layer')
tf.app.flags.DEFINE_integer('batch_size', 1, 'The batch size of testing')

FLAGS = tf.app.flags.FLAGS


def test1(dir, status='X2Y'):
    capture = cv2.VideoCapture(dir)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FLAGS.img_size)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FLAGS.img_size)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        _, content_img = capture.read()
        content = np.expand_dims(encode(content_img), 0)
        network = CycleGAN(FLAGS.batch_size, FLAGS.ngf, FLAGS.img_size, is_training=False, Norm=FLAGS.Norm)
        network.test(content.shape)
        sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.global_variables() if 'generator_net' in var.name or
                    'discriminator_net' in var.name or 'edge_net' in var.name]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, FLAGS.checkpoint)
        print("restored all")
        while True:
            if status == 'X2Y':
                out = sess.run(network.Ygenerated, feed_dict={network.testA: content})
            else:
                out = sess.run(network.Xgenerated, feed_dict={network.testB: content})
            # result = np.clip(out, 0, 255).astype(np.uint8)
            cv2.imshow("test", result)
            c = cv2.waitKey(1)
            if c == 27:
                break
            _, content_img = capture.read()
            content = np.expand_dims(encode(content_img), 0)
        capture.release()
        cv2.destroyAllWindows()


def test2(status='X2Y'):
    start = time.time()
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        content_img = cv2.resize(getImg(FLAGS.input), (FLAGS.img_size, FLAGS.img_size))
        content = np.expand_dims(encode(content_img), 0)
        network = CycleGAN(FLAGS.batch_size, FLAGS.ngf, FLAGS.img_size, is_training=True, use_E=FLAGS.USE_E,
                           Norm=FLAGS.Norm)
        network.test(content.shape)
        sess.run(tf.global_variables_initializer())
        var_list = [var for var in tf.global_variables() if
                    'generator_net' in var.name or 'discriminator_net' in var.name or 'edge_net' in var.name]
        saver = tf.train.Saver(var_list=var_list)
        saver.restore(sess, FLAGS.checkpoint)
        print("restored all")
        s = time.time()
        if status == 'X2Y':
            out = sess.run(network.Ygenerated, feed_dict={network.testA: content})
        else:
            out = sess.run(network.Xgenerated, feed_dict={network.testB: content})
        # result = np.clip(out[0], 0, 255).astype(np.uint8)
        print("Transform in {} s".format((time.time() - s)))
        saveImg(out[0], FLAGS.output)
        print("Finished all process in {} s".format(time.time() - start))


def check():
    reader = pywrap_tensorflow.NewCheckpointReader(FLAGS.checkpoint)
    var_to_shape_map = reader.get_variable_to_shape_map()
    for key in var_to_shape_map:
        if 'G_Model' in key:
            print("tensor_name: ", key)


if __name__ == '__main__':
    test2(FLAGS.status)
    # check()
