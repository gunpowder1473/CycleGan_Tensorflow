from GANNetwork import createNetwork
import tensorflow as tf
import common.common as cm


class BigGAN:
    def __init__(self, batch_size, ngf_G=1024, ngf_D=32, img_size=128, Norm='INSTANCE', is_training=True):
        self.ngf_G = ngf_G
        self.ngf_D = ngf_D
        self.Norm = Norm
        self.img_size = img_size
        self.batch_size = batch_size
        self.G = createNetwork.BigGenerator('G_Model', ngf_G, img_size, is_training=is_training)
        self.D = createNetwork.BigDiscriminator('D_Model', ngf_D, is_training=is_training)

    def discriminatorLoss(self, d_logits_real, d_logits_fake):
        real_loss = tf.reduce_mean(tf.nn.relu(1.0 - d_logits_real))
        fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + d_logits_fake))
        loss = real_loss + fake_loss
        return loss

    def generatorLoss(self, d_logits_fake):
        loss = -tf.reduce_mean(d_logits_fake)
        return loss

    def train(self, lambda1, lambda2):
        self.inputA = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size, self.img_size, 3),
                                     name='imgA')
        self.inputB = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, 1, 1, self.img_size),
                                     name='imgB')
        self.fake_rand = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size,
                                                                 self.img_size, 3), name='img2B')

        self.fake = self.G(self.inputB)

        self.real_logits = self.D(self.inputA)
        self.fake_logits = self.D(self.fake)
        self.fake_rand_logits = self.D(self.fake_rand)

        self.train_gen = cm.valid(self.fake)
        self.train_origin = cm.valid(self.inputA)

        with tf.name_scope('losses'):
            self.g_loss = self.generatorLoss(self.fake_logits)
            self.d_loss = self.discriminatorLoss(self.real_logits, self.fake_logits)
            self.firt_d_loss = self.discriminatorLoss(self.real_logits, self.fake_rand_logits)

    def test(self, batch_shape):
        self.test = tf.placeholder(dtype=tf.float32, shape=batch_shape,
                                   name='test')
        self.generated = cm.valid(self.G(self.test))
