import net.net as net
import tensorflow as tf


class Generator:
    def __init__(self, name, ngf, img_size=128, Norm='INSTANCE', is_training=True):
        self.ngf = ngf
        self.name = name
        self.reuse = False
        self.Norm = Norm
        self.img_size = img_size
        self.is_training = is_training

    def __call__(self, image):
        with tf.variable_scope('generator_net' + self.name, reuse=self.reuse):
            result = net.convLayer(image, self.ngf, 7, Norm=self.Norm, training=self.is_training, name='7x7_1',
                                   relu='RELU')
            result = net.convLayer(result, 2 * self.ngf, 3, strides=2, Norm=self.Norm, training=self.is_training,
                                   name='3x3_1', pad='SAME', relu='RELU')
            result = net.convLayer(result, 4 * self.ngf, 3, strides=2, Norm=self.Norm, training=self.is_training,
                                   name='3x3_2', pad='SAME', relu='RELU')
            if self.img_size == 128:
                for i in range(6):
                    result = net.residualBlock(result, 4 * self.ngf, 3, 'res' + '{:d}'.format(i), relu='RELU',
                                               pad='REFLECT', Norm=self.Norm, training=self.is_training)
            else:
                for i in range(9):
                    result = net.residualBlock(result, 4 * self.ngf, 3, 'res' + '{:d}'.format(i), relu='RELU',
                                               pad='REFLECT', Norm=self.Norm, training=self.is_training)
            result = net.transposeConv(result, 2 * self.ngf, 3, name='3x3_3', strides=2, pad='SAME', relu='RELU',
                                       bias=False, Norm=self.Norm, training=self.is_training)
            result = net.transposeConv(result, self.ngf, 3, name='3x3_4', strides=2, pad='SAME', relu='RELU',
                                       bias=False, Norm=self.Norm, training=self.is_training)
            result = net.convLayer(result, 3, 7, Norm=‘NOT’, training=self.is_training, relu=False, name='7x7_2')
            result = tf.tanh(result)
        self.reuse = True
        return result

class Discriminator:
    def __init__(self, name, Norm='INSTANCE', is_training=True):
        self.reuse = False
        self.Norm = Norm
        self.is_training = is_training
        self.name = name

    def __call__(self, image):
        with tf.variable_scope('discriminator_net' + self.name, reuse=self.reuse):
            result = net.convLayer(image, 64, 4, strides=2, name='4x4_1', relu=.2, Norm='NOT',
                                   training=self.is_training, pad='SAME')
            result = net.convLayer(result, 128, 4, strides=2, name='4x4_2', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result = net.convLayer(result, 256, 4, strides=2, name='4x4_3', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result = net.convLayer(result, 512, 4, strides=2, name='4x4_4', relu=.2, Norm=self.Norm,
                                   training=self.is_training, pad='SAME')
            result = net.convLayer(result, 1, 4, strides=1, name='4x4_5', relu=False, Norm='NOT',
                                   training=self.is_training, bias=True, pad='SAME')
        self.reuse = True
        return result


class Edge:
    def __init__(self, name, Norm='INSTANCE', is_training=True):
        self.reuse = False
        self.Norm = Norm
        self.is_training = is_training
        self.name = name

    def __call__(self, image):
        with tf.variable_scope('edge_net' + self.name, reuse=self.reuse):
            result = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], mode="REFLECT")
            pre = net.convLayer(result, 3, 3, strides=1, Norm=self.Norm, training=self.is_training,
                                name='Edge_P', pad='VALID', relu=False)
            result1 = net.convLayer(pre, 3, 3, strides=1, Norm=self.Norm, training=self.is_training,
                                    name='Edge_1', pad='VALID', relu='RELU')
            result2 = net.convLayer(pre, 3, 3, strides=1, Norm=self.Norm, training=self.is_training,
                                    name='Edge_2', pad='VALID', relu='RELU')
            result = net.convLayer(result1 + result2, 3, 3, strides=1, Norm='NOT', training=self.is_training,
                                   relu=False, name='Edge_3', pad='VALID')
            result = tf.tanh(result)
        self.reuse = True
        return result


class BigGenerator:
    def __init__(self, name, ngf=1024, img_size=128, is_training=True):
        self.ngf = ngf
        self.name = name
        self.reuse = False
        self.img_size = img_size
        self.is_training = is_training

    def __call__(self, image):
        with tf.variable_scope('generator_net' + self.name, reuse=self.reuse):
            result = net.transposeConv(image, self.ngf // 2, 4, pad='VALID', strides=1, Norm='NOT,SPECTRAL',
                                       training=self.is_training, name='deconv', relu=False, bias=False)
            result = net.residualBlockUp(result, self.ngf // 2, 3, 'resU1', pad='SAME', training=self.is_training)
            result = net.residualBlockUp(result, self.ngf // 4, 3, 'resU2', pad='SAME', training=self.is_training)
            result = net.residualBlockUp(result, self.ngf // 8, 3, 'resU3', pad='SAME', training=self.is_training)
            result = net.attention(result, self.ngf // 8, name='attentionG', is_training=self.is_training)
            result = net.residualBlockUp(result, self.ngf // 16, 3, 'resU4', pad='SAME', training=self.is_training)
            result = net.residualBlockUp(result, self.ngf // 32, 3, 'resU5', pad='SAME', training=self.is_training)
            result = net.batchNorm(result, self.is_training, name='bn1')
            result = tf.nn.leaky_relu(result)
            result = net.convLayer(result, 3, 3, Norm='NOT,SPECTRAL', training=self.is_training, relu=False, pad='SAME',
                                   name='G_Logit')
            result = tf.tanh(result)
        self.reuse = True
        return result


class BigDiscriminator:
    def __init__(self, name, ngf=32, is_training=True):
        self.reuse = False
        self.is_training = is_training
        self.name = name
        self.ngf = ngf

    def __call__(self, image):
        with tf.variable_scope('discriminator_net' + self.name, reuse=self.reuse):
            result = net.residualBlockDown(image, self.ngf * 2, 3, 'resD1', pad='SAME', training=self.is_training)
            result = net.residualBlockDown(result, self.ngf * 4, 3, 'resD2', pad='SAME', training=self.is_training)
            result = net.attention(result, self.ngf * 4, name='attentionD', is_training=self.is_training)
            result = net.residualBlockDown(result, self.ngf * 8, 3, 'resD3', pad='SAME', training=self.is_training)
            result = net.residualBlockDown(result, self.ngf * 16, 3, 'resD4', pad='SAME', training=self.is_training)
            result = net.residualBlockDown(result, self.ngf * 16, 3, 'resD5', pad='SAME', training=self.is_training)
            result = net.residualBlockDown(result, self.ngf * 32, 3, 'resD6', pad='SAME', training=self.is_training)
            result = net.convLayer(result, 1, 4, Norm='NOT,SPECTRAL', training=self.is_training, relu=False,
                                   name='D_Logit', pad='VALID')
            result = tf.squeeze(result, axis=[1, 2])
        self.reuse = True
        return result
