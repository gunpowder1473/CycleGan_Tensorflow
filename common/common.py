import tensorflow as tf
import random
import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from scipy.stats import truncnorm


def torch_decay(learning_rate, global_step, decay_rate, name=None):
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    with tf.name_scope(name, "ExponentialDecay", [learning_rate, global_step, decay_rate]) as name:
        learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = tf.cast(global_step, dtype)
        decay_rate = tf.cast(decay_rate, dtype)
        return learning_rate / (1 + global_step * decay_rate)


def exponential_decay(learning_rate, global_step, start_step, decay_steps, decay_rate):
    return tf.train.exponential_decay(learning_rate,
                                      tf.maximum(global_step - start_step, 0),
                                      decay_steps,
                                      decay_rate)


def linear_decay(learning_rate, global_step, start_step, end_step, name=None):
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    with tf.name_scope(name, "LinearDecay", [learning_rate, global_step, start_step, end_step]) as name:
        learning_rate = tf.convert_to_tensor(learning_rate, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = tf.cast(global_step, dtype)
        end_step = tf.cast(end_step, dtype)
        return (end_step - global_step) * learning_rate / (end_step - start_step)


def valid(img):
    return tf.image.convert_image_dtype((img + 1.0) / 2.0, tf.uint8)


def encode(img):
    return img / 127.5 - 1.0


def getFiles(imgpath, name):
    paths = []

    def walk(path):
        files = os.listdir(path)
        for x in files:
            p = os.path.join(path, x)
            if os.path.isdir(p):
                walk(p)
            else:
                paths.append(p)

    walk(imgpath)
    print(name + 'Load Finished')
    return paths


def saveImg(img, outpath):
    img = np.clip(img, 0, 255).astype(np.uint8)
    # files = os.listdir(outpath)
    # num = len(files)
    plt.imsave(os.path.join(outpath), img)


def getImg(inpath):
    img = plt.imread(inpath)
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    return img


def clip(x):
    return tf.clip_by_value(x, 0, 1)


def resizeTo(img, resize_L=800, resize_U=1800):
    height, width = img.shape[0], img.shape[1]
    if height < width:
        ratio = height / resize_L
        long_side = round(width / ratio)
        if long_side >= resize_U:
            long_side = resize_U
            resize_shape = (resize_L, long_side)
    else:
        ratio = width / resize_L
        long_side = round(height / ratio)
        if long_side >= resize_U:
            long_side = resize_U
        resize_shape = (long_side, resize_L)
    return cv2.resize(img, resize_shape, interpolation=cv2.INTER_CUBIC)


def imgRandomCrop(src, resize_L=800, resize_U=1800, crop=128):
    img = getImg(src)
    img = resizeTo(img, resize_L, resize_U)

    offset_h = random.randint(0, (img.shape[0] - crop))
    offset_w = random.randint(0, (img.shape[1] - crop))

    img = img[offset_h:offset_h + crop, offset_w:offset_w + crop, :]
    return img


class imgPool:
    def __init__(self, size):
        self.size = size
        self.images = []

    def __call__(self, input):
        if self.size == 0:
            return input
        if len(self.images) < self.size:
            self.images.append(input)
            return input
        else:
            p = random.random()
            if p > 0.5:
                choosed = random.randrange(0, self.size)
                img = self.images[choosed].copy()
                self.images[choosed] = input.copy()
                return img
            else:
                return input


def getEdge(img):
    dest = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dest = cv2.GaussianBlur(dest, (3, 3), 0)
    x = cv2.Sobel(dest, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(dest, cv2.CV_16S, 0, 1)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dest = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    dest = np.dstack((dest, dest, dest))
    dest = 255 - dest
    return dest


def getTruncatedNormal(mean=0, sd=0.5):
    return truncnorm((sd * -2 - mean) / sd, (sd * 2 - mean) / sd, loc=mean, scale=sd)
