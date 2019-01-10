from GANNetwork import createNetwork
import tensorflow as tf
import common.common as cm


class CycleGAN:
    def __init__(self, batch_size, ngf=64, img_size=128, Norm='INSTANCE', is_training=True, use_E=False):
        self.ngf = ngf
        self.Norm = Norm
        self.use_E = use_E
        self.img_size = img_size
        self.batch_size = batch_size
        self.G = createNetwork.Generator('G_Model', ngf, img_size, is_training=is_training, Norm=Norm)
        self.D_Y = createNetwork.Discriminator('D_Y_Model', is_training=is_training, Norm=Norm)
        self.F = createNetwork.Generator('F_Model', ngf, img_size, is_training=is_training, Norm=Norm)
        self.D_X = createNetwork.Discriminator('D_X_Model', is_training=is_training, Norm=Norm)
        self.E = createNetwork.Edge('Edge_Model', is_training=is_training, Norm=Norm)

    def discriminatorLoss(self, D, y, fake_y):
        error_real = tf.reduce_mean(tf.squared_difference(D(y), 1))
        error_fake = tf.reduce_mean(tf.square(D(fake_y)))
        loss = (error_real + error_fake) / 2
        return loss

    def generatorLoss(self, D, fake_y):
        loss = tf.reduce_mean(tf.squared_difference(D(fake_y), 1))
        return loss

    # def discriminatorLoss(self, D, y, fake_y):
    #     error_real = tf.reduce_mean(tf.nn.relu(1 - D(y)))
    #     error_fake = tf.reduce_mean(tf.nn.relu(1 + D(fake_y)))
    #     loss = (error_real + error_fake)
    #     return loss
    #
    # def generatorLoss(self, D, fake_y):
    #     loss = -tf.reduce_mean(D(fake_y))
    #     return loss

    def cycleConsistencyLoss(self, G, F, x, y, lambda1, lambda2):
        forward_loss = tf.reduce_mean(tf.abs(F(G(x)) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(F(y)) - y))
        loss = lambda1 * forward_loss + lambda2 * backward_loss
        return loss

    def cycleConsistencyLoss2(self, G, F, x, y, lambda1, lambda2):
        forward_loss = tf.reduce_mean(tf.abs(F(self.E(G(self.E(x)))) - x))
        backward_loss = tf.reduce_mean(tf.abs(G(self.E(F(self.E(y)))) - y))
        loss = lambda1 * forward_loss + lambda2 * backward_loss
        return loss

    def identityLoss(self, G, F, x, y, lambda1, lambda2):
        loss = lambda1 * tf.reduce_mean(tf.abs(G(y) - y)) + lambda2 * tf.reduce_mean(tf.abs(F(x) - x))
        return loss

    def identityLoss2(self, G, F, x, y, lambda1, lambda2):
        loss = lambda1 * tf.reduce_mean(tf.abs(G(self.E(y)) - y)) + lambda2 * tf.reduce_mean(
            tf.abs(F(self.E(x)) - x))
        return loss

    def train(self, lambda1, lambda2, pool_A, pool_B):
        self.inputA = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size, self.img_size, 3),
                                     name='imgA')
        self.inputB = tf.placeholder(dtype=tf.float32, shape=(self.batch_size, self.img_size, self.img_size, 3),
                                     name='imgB')

        if self.use_E:
            self.inputA_E = self.E(self.inputA)
            self.inputB_E = self.E(self.inputB)
            self.fake_y = self.G(self.inputA_E)
            self.fake_x = self.F(self.inputB_E)
            self.genXE = cm.valid(self.E(self.fake_x))
            self.genYE = cm.valid(self.E(self.fake_y))
            self.reconstruct_x = cm.valid(self.F(self.E(self.G(self.E(self.inputA)))))
            self.reconstruct_y = cm.valid(self.G(self.E(self.F(self.E(self.inputB)))))
            self.inputBE = cm.valid(self.inputB_E)
            self.inputAE = cm.valid(self.inputA_E)
        else:
            self.fake_y = self.G(self.inputA)
            self.fake_x = self.F(self.inputB)
            self.genXE = cm.valid(self.fake_x)
            self.genYE = cm.valid(self.fake_y)
            self.reconstruct_x = cm.valid(self.F(self.G(self.inputA)))
            self.reconstruct_y = cm.valid(self.G(self.F(self.inputB)))

        self.train_gen_x = cm.valid(self.fake_x)
        self.train_gen_y = cm.valid(self.fake_y)
        self.train_x = cm.valid(self.inputA)
        self.train_y = cm.valid(self.inputB)

        with tf.name_scope('losses'):
            self.G_gan_loss = self.generatorLoss(self.D_Y, self.fake_y)
            self.F_gan_loss = self.generatorLoss(self.D_X, self.fake_x)
            if self.use_E:
                self.cycle_loss = self.cycleConsistencyLoss2(self.G, self.F, self.inputA, self.inputB, lambda1, lambda2)
                self.identity_loss = self.identityLoss2(self.G, self.F, self.inputA, self.inputB, lambda1, lambda2)
            else:
                self.cycle_loss = self.cycleConsistencyLoss(self.G, self.F, self.inputA, self.inputB, lambda1, lambda2)
                self.identity_loss = self.identityLoss(self.G, self.F, self.inputA, self.inputB, lambda1, lambda2)
            self.exter_loss = self.cycle_loss + self.identity_loss
            self.Gan_loss = self.F_gan_loss + self.G_gan_loss + self.cycle_loss + self.identity_loss
            self.D_Y_loss = self.discriminatorLoss(self.D_Y, self.inputB, pool_B(self.fake_y))
            self.D_X_loss = self.discriminatorLoss(self.D_X, self.inputA, pool_A(self.fake_x))
            self.D_loss = self.D_X_loss + self.D_Y_loss

    def test(self, batch_shape):
        self.testA = tf.placeholder(dtype=tf.float32, shape=batch_shape,
                                    name='testA')
        self.testB = tf.placeholder(dtype=tf.float32, shape=batch_shape,
                                    name='testB')
        if self.use_E:
            self.Ygenerated = cm.valid(self.G(self.E(self.testA)))
            self.Xgenerated = cm.valid(self.F(self.E(self.testB)))
        else:
            self.Ygenerated = cm.valid(self.G(self.testA))
            self.Xgenerated = cm.valid(self.F(self.testB))
