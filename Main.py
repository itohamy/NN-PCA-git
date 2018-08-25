# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
from data_provider import DataProvider
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Plots import open_figure, PlotImages
from Network import Network

batch_size = 64  # Number of samples in each batch
epoch_num = 1    # Number of epochs to train the network
lr = 0.001        # Learning rate


def train():

    # Load data (frames) from video
    video_name = "movies/BG.mp4"
    data = DataProvider(video_name)

    device = '/cpu:0' # '/gpu:0'
    with tf.device(device):
        # build the network
        net = Network()

        # calculate the number of batches per epoch
        batch_per_ep = data.train_size // batch_size
        # print('Data size: ', data.train_size, ' Num of epochs: ', epoch_num, ' Batches per epoch: ', batch_per_ep)

        ae_inputs = tf.placeholder(tf.float32, (batch_size, 128, 128, 1))  # input to the network
        ae_outputs = net.autoencoder(ae_inputs) # net.build(ae_inputs) or net.autoencoder(ae_inputs) # create the Autoencoder network

        # calculate the loss and optimize the network
        loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # initialize the network
        init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epoch_num):  # epochs loop
            for i in range(10000):  # batches loop
                # read a batch -> batch_img
                batch_img = data.next_batch(batch_size, 'train')
                # print("batch size: ", batch_img.shape)
                _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
                if not i % 10:
                    print('Epoch {0}: Iteration: {1} Loss: {2:.5f}'.format((ep + 1), i, c))

        # test the trained network
        batch_img = data.next_batch(batch_size, 'test')
        test_results = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]

        # plot the reconstructed images and their ground truths (inputs)
        imgs = []
        imgs_test = []
        titles = []
        for i in range(30):
            imgs.append(batch_img[i, ..., 0])
            imgs_test.append(test_results[i,...,0])
            titles.append('')
        fig1 = open_figure(1, 'Original Images', (10, 3))
        PlotImages(1, 3, 10, 1, imgs, titles, 'gray', axis=True, colorbar=False)
        fig2 = open_figure(2, 'Test Results', (10, 3))
        PlotImages(2, 3, 10, 1, imgs_test, titles, 'gray', axis=True, colorbar=False)
        fig1.savefig('f1.png')
        fig2.savefig('f2.png')


if __name__ == '__main__':
    train()