# coding=utf-8
import tensorflow as tf
from GANNetwork.cycleGAN import CycleGAN
from common.common import getImg, torch_decay, getFiles, saveImg, imgPool, encode, linear_decay, getEdge
import numpy as np
import random
import threading, os, time, cv2

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('imgA', "horse2zebra/trainA",
                           'The directory that A pictures are saved')
tf.app.flags.DEFINE_string('imgB', "horse2zebra/trainB",
                           'The directory that B picture are saved')
tf.app.flags.DEFINE_string('imgC',
                           "orange.jpg",
                           'The directory that validate pictures are saved')
tf.app.flags.DEFINE_string('val_out', "test/", 'The directory that validate pictures are saved')
tf.app.flags.DEFINE_string('checkpoint', "checkout/",
                           'The directory that trained network will be saved')
tf.app.flags.DEFINE_string('Norm', 'BATCH', 'Choose to use Batchnorm or instanceNorm')
tf.app.flags.DEFINE_bool('USE_E', False, 'Choose to use Edge or not')
tf.app.flags.DEFINE_float('learning_rate', 2e-4, 'The init learning rate')
tf.app.flags.DEFINE_float('decay', 1e-6, 'The init learning rate decay')
tf.app.flags.DEFINE_integer('start_step', 100000, 'The start step for linear decay')
tf.app.flags.DEFINE_integer('end_step', 200000, 'The end step for linear decay')
tf.app.flags.DEFINE_integer('max_to_keep', 10, 'The maximum ckpt num')
tf.app.flags.DEFINE_integer('summary_iter', 10, 'The steps per summary')
tf.app.flags.DEFINE_integer('save_iter', 200, 'The steps per save')
tf.app.flags.DEFINE_integer('val_iter', 400, 'The steps per validated')
tf.app.flags.DEFINE_integer('batch_size', 1, 'The batch size of training')
tf.app.flags.DEFINE_float('lambda1', 10.0, 'The weight of forward cycle loss')
tf.app.flags.DEFINE_float('lambda2', 10.0, 'The weight of backward cycle loss')
tf.app.flags.DEFINE_integer('ngf', 64, 'The number of gen filters in first conv layer')
tf.app.flags.DEFINE_integer('img_size', 256, 'The size of input img')

FLAGS = tf.app.flags.FLAGS


def generateBatch(folder, batch_shape):
    files = getFiles(folder)
    while True:
        try:
            batch = np.zeros(batch_shape, dtype=np.float32)
            choosed = random.sample(files, batch_shape[0])
            for i, s in enumerate(choosed):
                batch[i] = cv2.resize(getImg(s), (FLAGS.img_size, FLAGS.img_size))
                batch[i] = encode(batch[i])
            yield batch
        except:
            return


def train():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # val_img = getImg(FLAGS.imgC)
    # val = np.expand_dims(encode(val_img), 0)
    # val_batch_shape = val.shape

    with tf.Graph().as_default(), tf.Session(config=config) as sess:
        tf.logging.set_verbosity(tf.logging.INFO)
        queue_inputA = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
        queue_inputB = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
        queue = tf.FIFOQueue(capacity=100, dtypes=[tf.float32, tf.float32],
                             shapes=[[FLAGS.img_size, FLAGS.img_size, 3], [FLAGS.img_size, FLAGS.img_size, 3]])
        enqueue_op = queue.enqueue_many([queue_inputA, queue_inputB])
        dequeue_op = queue.dequeue()
        imgA_batch_op, imgB_batch_op = tf.train.batch(dequeue_op, batch_size=FLAGS.batch_size, capacity=100)

        with tf.device('/device:CPU:0'):
            global_step1 = tf.Variable(0, trainable=False)
            learning_rate1 = tf.where(tf.greater_equal(global_step1, FLAGS.start_step),
                                      linear_decay(FLAGS.learning_rate, global_step1, FLAGS.start_step, FLAGS.end_step),
                                      FLAGS.learning_rate)
            # learning_rate1 = torch_decay(FLAGS.learning_rate, global_step1, FLAGS.decay)
            opt1 = tf.train.AdamOptimizer(learning_rate1, beta1=0.5)
            # opt1 = tf.train.RMSPropOptimizer(learning_rate1)
            global_step2 = tf.Variable(0, trainable=False)
            learning_rate2 = tf.where(tf.greater_equal(global_step2, FLAGS.start_step),
                                      linear_decay(FLAGS.learning_rate, global_step2, FLAGS.start_step, FLAGS.end_step),
                                      FLAGS.learning_rate)
            # learning_rate2 = torch_decay(FLAGS.learning_rate, global_step2, FLAGS.decay)
            opt2 = tf.train.AdamOptimizer(learning_rate2, beta1=0.5)
            # opt2 = tf.train.RMSPropOptimizer(learning_rate2)
            global_step3 = tf.Variable(0, trainable=False)
            learning_rate3 = tf.where(tf.greater_equal(global_step3, FLAGS.start_step),
                                      linear_decay(FLAGS.learning_rate, global_step3, FLAGS.start_step, FLAGS.end_step),
                                      FLAGS.learning_rate)
            # learning_rate3 = torch_decay(FLAGS.learning_rate, global_step3, FLAGS.decay)
            opt3 = tf.train.AdamOptimizer(learning_rate3, beta1=0.5)

        fake_X = imgPool(50)
        fake_Y = imgPool(50)

        net = CycleGAN(FLAGS.batch_size, FLAGS.ngf, FLAGS.img_size, FLAGS.Norm, use_E=FLAGS.USE_E)
        net.train(FLAGS.lambda1, FLAGS.lambda2, fake_X, fake_Y)

        if FLAGS.USE_E is True:
            var_list_1 = [var for var in tf.trainable_variables() if
                        'G_Model' in var.name or 'F_Model' in var.name or 'Edge_Model' in var.name]
        else:
            var_list_1 = [var for var in tf.trainable_variables() if
                          'G_Model' in var.name or 'F_Model' in var.name]
        var_list_2 = [var for var in tf.trainable_variables() if 'D_X_Model' in var.name]
        var_list_3 = [var for var in tf.trainable_variables() if 'D_Y_Model' in var.name]

        if "BATCH" in FLAGS.Norm:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_1 = opt1.minimize(net.Gan_loss, global_step1, var_list_1)
                train_op_2 = opt2.minimize(net.D_X_loss, global_step2, var_list_2)
                train_op_3 = opt3.minimize(net.D_Y_loss, global_step3, var_list_3)
                train_op = tf.group([train_op_1, train_op_2, train_op_3])
        else:
            train_op_1 = opt1.minimize(net.Gan_loss, global_step1, var_list_1)
            train_op_2 = opt2.minimize(net.D_X_loss, global_step2, var_list_2)
            train_op_3 = opt3.minimize(net.D_Y_loss, global_step3, var_list_3)
            train_op = tf.group([train_op_1, train_op_2, train_op_3])
        saver = tf.train.Saver(max_to_keep=FLAGS.max_to_keep)

        with tf.device('/device:CPU:0'):
            with tf.name_scope('summary'):
                tf.summary.scalar('learning_rate', learning_rate1)
                with tf.name_scope('gen_img'):
                    tf.summary.image('gen_x', net.train_gen_x)
                    tf.summary.image('gen_y', net.train_gen_y)
                    tf.summary.image('original_x', net.inputA)
                    tf.summary.image('original_y', net.inputB)
                    if FLAGS.USE_E:
                        tf.summary.image('original_x_edge', net.inputAE)
                        tf.summary.image('original_y_edge', net.inputBE)
                        tf.summary.image('gen_x_edge', net.genXE)
                        tf.summary.image('gen_y_edge', net.genYE)
                    tf.summary.image('reconstruct_x', net.reconstruct_x)
                    tf.summary.image('reconstruct_y', net.reconstruct_y)
                with tf.name_scope('loss'):
                    tf.summary.scalar('Gan_loss', net.Gan_loss)
                    tf.summary.scalar('D_loss', net.D_loss)
                    tf.summary.scalar('D_X_loss', net.D_X_loss)
                    tf.summary.scalar('D_Y_loss', net.D_Y_loss)
                    tf.summary.scalar('Cycle_loss', net.cycle_loss)
                    tf.summary.scalar('X2Y_Gen_loss', net.G_gan_loss)
                    tf.summary.scalar('Y2X_Gen_loss', net.F_gan_loss)
                    tf.summary.scalar('Identify_Loss', net.identity_loss)
                summary_op = tf.summary.merge_all()

        coord = tf.train.Coordinator()

        def enqueue(sess):
            imgA = generateBatch(FLAGS.imgA, (FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
            imgB = generateBatch(FLAGS.imgB, (FLAGS.batch_size, FLAGS.img_size, FLAGS.img_size, 3))
            while not coord.should_stop():
                imgA_batch = next(imgA)
                imgB_batch = next(imgB)
                try:
                    sess.run(enqueue_op, feed_dict={queue_inputA: imgA_batch, queue_inputB: imgB_batch})
                except:
                    print("The img reading thread is end")

        log_path = os.path.join(FLAGS.checkpoint, 'log')
        summary_writer = tf.summary.FileWriter(log_path, sess.graph)
        sess.run(tf.global_variables_initializer())

        if os.path.exists(os.path.join(FLAGS.checkpoint, 'checkpoint')):
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))

        with tf.device('/device:CPU:0'):
            iteration = global_step1.eval() + 1

        enqueue_thread = threading.Thread(target=enqueue, args=[sess])
        enqueue_thread.isDaemon()
        enqueue_thread.start()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        while True:
            try:
                start = time.time()

                imgA_batch = sess.run(imgA_batch_op)
                imgB_batch = sess.run(imgB_batch_op)
                # output = sess.run({'train': train_op, 'global_step': global_step1},
                #     feed_dict={net.inputA: imgA_batch, net.inputB: imgB_batch})
                output = sess.run(
                    {'fake_X': net.fake_x, 'fake_Y': net.fake_y, 'train': train_op, 'global_step': global_step1,
                     'learning_rate': learning_rate1, 'Gan_Loss': net.Gan_loss, 'Cycle_Loss': net.cycle_loss,
                     'G_Gen_Loss': net.G_gan_loss, 'D_X_Loss': net.D_X_loss, 'D_Y_Loss': net.D_Y_loss,
                     'F_Gen_Loss': net.F_gan_loss, 'D_Loss': net.D_loss, 'Identity_Loss': net.identity_loss,
                     'summary': summary_op},
                    feed_dict={net.inputA: imgA_batch, net.inputB: imgB_batch})
            except Exception as e:
                coord.request_stop(e)
                print("Get error as {} , need reload".format(e))
                if os.path.exists(os.path.join(FLAGS.checkpoint, 'checkpoint')):
                    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint))
                    print("Restoring checkpoint")
                    continue
                else:
                    print("No checkpoint")
                    break

            if iteration % FLAGS.summary_iter == 0:
                summary_writer.add_summary(output['summary'], output['global_step'])

            if iteration % FLAGS.save_iter == 0:
                save_path = saver.save(sess, os.path.join(FLAGS.checkpoint, 'model.ckpt'), output['global_step'])
                print("Model saved in file: %s" % save_path)

            if iteration % FLAGS.val_iter == 0:
                pass
                # val_out = val_sess.run(network.Xgenerated, feed_dict={network.testB: val})
                # result = np.clip(val_out[0], 0, 255).astype(np.uint8)
                # saveImg(result, os.path.join(FLAGS.val_out, 'val_' + str(output['global_step']) + '.jpg'))
                # print("Validate done")

            print(
                "At Step {},with learning_rate is {:.7f}, get Gan_Loss {:.2f}, D_loss {:.2f}, Cycle_Loss {:.2f}, D_X_Loss {:.2f},"
                " D_Y_Loss {:.2f}, X2Y_Gen_Loss {:.2f}, Y2X_Gen_Loss {:.2f}, Identity_Loss {:.2f}, cost {:.2f}s".
                    format(
                    output['global_step'],
                    output['learning_rate'],
                    output['Gan_Loss'],
                    output['D_Loss'],
                    output['Cycle_Loss'],
                    output['D_X_Loss'],
                    output['D_Y_Loss'],
                    output['G_Gen_Loss'],
                    output['F_Gen_Loss'],
                    output['Identity_Loss'],
                    time.time() - start))

            if (output['global_step'] >= FLAGS.end_step):
                break

            iteration += 1
        print('done')
        save_path = saver.save(sess, os.path.join(FLAGS.checkpoint, 'model.ckpt'), output['global_step'])
        print("Model saved in file: %s" % save_path)
        coord.request_stop()
        queue.close()
        coord.join(threads)
        print("All end")


if __name__ == '__main__':
    train()
