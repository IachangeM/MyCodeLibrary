import os
from datetime import datetime

import numpy as np
from scipy.misc import imsave
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import tensorlayer as tl


class Model:
    def __init__(self, args, data):
        self.args = args
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_test = data['X_test']
        self.y_test = data['y_test']

    def build_net(self):

        height = 512
        width = 512
        initializer = tf.contrib.layers.xavier_initializer_conv2d()

        self.X = tf.placeholder(tf.float32, shape=(None, height, width, 1), name="X")
        self.y = tf.placeholder(tf.float32, shape=(None, height, width, 1), name="y")

        # ================== Contracting Path ===================
        with tf.name_scope('u_net'):
            self.conv1 = tf.layers.conv2d(inputs=self.X, filters=64, kernel_size=(3, 3), padding='same',
                                          activation=tf.nn.relu, kernel_initializer=initializer)

            self.conv1 = tf.layers.conv2d(self.conv1, 64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv1.2', kernel_initializer=initializer)

            self.drop1 = tf.nn.dropout(self.conv1, keep_prob=self.args.keep_prob, name='drop1')

            self.max_pool1 = tf.layers.max_pooling2d(self.drop1, pool_size=(2, 2), strides=(2, 2), padding='same',
                                                     name='max_pool1')

            self.conv2 = tf.layers.conv2d(self.max_pool1, 128, kernel_size=(3, 3), padding='same',
                                          activation=tf.nn.relu, name='conv2.1', kernel_initializer=initializer)

            self.conv2 = tf.layers.conv2d(self.conv2, 128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv2.2', kernel_initializer=initializer)

            self.drop2 = tf.nn.dropout(self.conv2, keep_prob=self.args.keep_prob, name='drop2')

            self.max_pool2 = tf.layers.max_pooling2d(self.drop2, pool_size=(2, 2), strides=(2, 2), padding='same',
                                                     name='max_pool2')

            self.conv3 = tf.layers.conv2d(self.max_pool2, 256, kernel_size=(3, 3), padding='same',
                                          activation=tf.nn.relu, name='conv3.1', kernel_initializer=initializer)

            self.conv3 = tf.layers.conv2d(self.conv3, strides=(1, 1), filters=256, kernel_size=(3, 3), padding='same',
                                          activation=tf.nn.relu, name='conv3.2', kernel_initializer=initializer)

            self.drop3 = tf.nn.dropout(self.conv3, keep_prob=self.args.keep_prob, name='drop3')

            self.max_pool3 = tf.layers.max_pooling2d(self.drop3, pool_size=(2, 2), strides=(2, 2), padding='same',
                                                     name='max_pool3')

            self.conv4 = tf.layers.conv2d(self.max_pool3, 512, kernel_size=(3, 3), padding='same',
                                          activation=tf.nn.relu, name='conv4.1', kernel_initializer=initializer)

            self.conv4 = tf.layers.conv2d(self.conv4, 512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv4.2', kernel_initializer=initializer)

            self.drop4 = tf.nn.dropout(self.conv4, keep_prob=self.args.keep_prob, name='drop4')

            self.max_pool4 = tf.layers.max_pooling2d(self.drop4, pool_size=(2, 2), strides=(2, 2), padding='same',
                                                     name='max_pool4')

            self.conv5 = tf.layers.conv2d(self.max_pool4, 1024, kernel_size=(3, 3), padding='same',
                                          activation=tf.nn.relu, name='conv5.1', kernel_initializer=initializer)

            self.conv5 = tf.layers.conv2d(self.conv5, 1024, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv5.2', kernel_initializer=initializer)

            self.drop5 = tf.nn.dropout(self.conv5, keep_prob=self.args.keep_prob, name='drop5')

            # ================== Expanding Path ===================


            self.deconv1 = tf.layers.conv2d_transpose(self.drop5, 512, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same',
                                                      activation=tf.nn.relu, name='upconv1.1',
                                                      kernel_initializer=initializer)

            self.deconv1 = tf.concat([self.deconv1, self.conv4], 3, name='concat1')

            self.conv6 = tf.layers.conv2d(self.deconv1, 512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv6.1', kernel_initializer=initializer)

            self.conv6 = tf.layers.conv2d(self.conv6, 512, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv6.2', kernel_initializer=initializer)

            self.drop6 = tf.nn.dropout(self.conv6, keep_prob=self.args.keep_prob, name='drop6')

            self.deconv2 = tf.layers.conv2d_transpose(self.drop6, 256, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same',
                                                      activation=tf.nn.relu, name='upconv2.1',
                                                      kernel_initializer=initializer)

            self.deconv2 = tf.concat([self.deconv2, self.conv3], 3, name='concat2')

            self.conv7 = tf.layers.conv2d(self.deconv2, 256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv7.1', kernel_initializer=initializer)

            self.conv7 = tf.layers.conv2d(self.conv7, 256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv7.2', kernel_initializer=initializer)

            self.drop7 = tf.nn.dropout(self.conv7, keep_prob=self.args.keep_prob, name='drop7')

            self.deconv3 = tf.layers.conv2d_transpose(self.drop7, 128, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same',
                                                      activation=tf.nn.relu, name='upconv3.1',
                                                      kernel_initializer=initializer)

            self.deconv3 = tf.concat([self.deconv3, self.conv2], 3, name='concat3')

            self.conv8 = tf.layers.conv2d(self.deconv3, 128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv8.1', kernel_initializer=initializer)

            self.conv8 = tf.layers.conv2d(self.conv8, 128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv8.2', kernel_initializer=initializer)

            self.drop8 = tf.nn.dropout(self.conv8, keep_prob=self.args.keep_prob, name='drop8')

            self.deconv4 = tf.layers.conv2d_transpose(self.drop8, 64, kernel_size=(2, 2), strides=(2, 2),
                                                      padding='same',
                                                      activation=tf.nn.relu, name='upconv4.1',
                                                      kernel_initializer=initializer)

            self.deconv4 = tf.concat([self.deconv4, self.conv1], 3, name='concat4')

            self.conv9 = tf.layers.conv2d(self.deconv4, 64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv9.1', kernel_initializer=initializer)

            self.conv9 = tf.layers.conv2d(self.conv9, 64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu,
                                          name='conv9.2', kernel_initializer=initializer)

            self.drop9 = tf.nn.dropout(self.conv9, keep_prob=self.args.keep_prob, name='drop9')

            self.logits = tf.layers.conv2d(self.drop9, 1, kernel_size=(1, 1), padding='same', activation=tf.nn.sigmoid,
                                           name='logits', kernel_initializer=initializer)

        with tf.name_scope('loss'):
            self.loss = 1 - tl.cost.dice_coe(self.logits, self.y, axis=[0, 1, 2, 3])
            self.loss = tf.reduce_mean(self.loss, name='loss')

        with tf.name_scope('train'):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.args.learning_rate)
            self.training_op = self.optimizer.minimize(self.loss)

    def train(self):

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        if self.args.load_checkpoint is not None:
            self.load(self.args.load_checkpoint)

        print('Initializing training...')

        n_epochs = self.args.n_epochs
        batch_size = self.args.batch_size
        best_loss = np.infty
        max_checks_without_progress = self.args.early_stopping_max_checks
        checks_without_progress = 0

        for epoch in range(n_epochs):
            X_train, y_train = self.unison_shuffled_copies(self.X_train, self.y_train)
            for iteration in range(X_train.shape[0] // batch_size):
                X_batch, y_batch = self.data_provider(X_train, y_train, iteration)
                self.sess.run(self.training_op, feed_dict={self.X: X_batch, self.y: y_batch})
            if epoch % 10 == 0:
                loss_train = []
                loss_val = []
                for j in range(self.X_train.shape[0]):
                    X_val_batch, y_val_batch = self.data_provider(self.X_train, self.y_train, j)
                    loss_val.append(
                        self.loss.eval(session=self.sess, feed_dict={self.X: X_val_batch, self.y: y_val_batch}))
                    X_train_batch, y_train_batch = self.data_provider(X_train, y_train, j)
                    loss_train.append(
                        self.loss.eval(session=self.sess, feed_dict={self.X: X_train_batch, self.y: y_train_batch}))
                loss_train = np.mean(loss_train)
                loss_val = np.mean(loss_val)
                if loss_val < best_loss:
                    self.save(epoch)
                    best_loss = loss_val
                    checks_without_progress = 0
                else:
                    checks_without_progress += 1
                    if checks_without_progress > max_checks_without_progress:
                        print("Early stopping!")
                        self.save(epoch)
                        break
                print(epoch, "Train Loss:", loss_train, "Validation loss:", loss_val)

    def infer(self):
        self.sess = tf.Session()
        self.load(self.args.load_checkpoint)
        dice_list = []
        roc_list = []
        for iteration in range(self.X_test.shape[0] // self.args.batch_size):
            X_batch, y_batch = self.data_provider(self.X_test, self.y_test, iteration)
            output = self.sess.run(self.logits, feed_dict={self.X: X_batch})
            output_mask = np.squeeze((output > 0.5).astype(dtype=np.float32))
            y_batch = np.squeeze(y_batch)
            y_batch_mask = y_batch > 0.5
            y_batch_mask = y_batch_mask.astype(np.int64)
            dice_list.append(np.sum(output_mask[y_batch_mask == 1.0])*2.0 / (np.sum(output_mask) + np.sum(y_batch_mask)))
            roc_list.append(roc_auc_score(y_true = y_batch_mask.flatten(), y_score = np.squeeze(output).flatten()))
            # imsave('../data/output/{}.png'.format(iteration), np.squeeze(output))
            # imsave('../data/output/{}_anno.png'.format(iteration), np.squeeze(y_batch))
        print((sum(dice_list) / len(dice_list)), sum(roc_list)/len(roc_list))

    def data_provider(self, X, y, iteration):
        begin = self.args.batch_size * iteration
        end = self.args.batch_size * (iteration + 1)
        return X[begin:end, :, :, :], y[begin:end, :, :, :]

    def unison_shuffled_copies(self, X, y):
        p = np.random.permutation(X.shape[0])
        return X[p, :, :, :], y[p, :, :, :]

    def save(self, epoch):
        print('[*] Saving checkpoint ....')
        model_name = 'model_{}_epoch_{}.ckpt'.format(datetime.now().strftime("%d:%H:%M:%S"), epoch)
        self.saver = tf.train.Saver()
        save_path = self.saver.save(self.sess, os.path.join(self.args.saved_model_directory, model_name))
        print('[*] Checkpoint saved in file {}'.format(save_path))

    def load(self, model_name):
        print(" [*] Loading checkpoint...")
        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, os.path.join(self.args.saved_model_directory, model_name))
