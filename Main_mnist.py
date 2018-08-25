# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
from video_to_frames import extractImages
from PIL import Image
#from resizeimage import resizeimage
from Plots import open_figure, PlotImages

batch_size = 64  # Number of samples in each batch
epoch_num = 1   # Number of epochs to train the network
lr = 0.001        # Learning rate


def resize_batch(imgs):
    # A function to resize a batch of MNIST images to (32, 32)
    # Args:
    #   imgs: a numpy array of size [batch_size, 28 X 28].
    # Returns:
    #   a numpy array of size [batch_size, 32, 32].
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs


def autoencoder(inputs):
    # encoder
    # 32 x 32 x 1   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  8 x 8 x 16
    # 8 x 8 x 16    ->  2 x 2 x 8
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    # decoder
    # 2 x 2 x 8    ->  8 x 8 x 16
    # 8 x 8 x 16   ->  16 x 16 x 32
    # 16 x 16 x 32  ->  32 x 32 x 1
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net


def train():

    # read MNIST dataset
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)

    # calculate the number of batches per epoch
    batch_per_ep = mnist.train.num_examples // batch_size
    print(batch_per_ep, mnist.train.num_examples)

    ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))  # input to the network (MNIST images)
    ae_outputs = autoencoder(ae_inputs)  # create the Autoencoder network

    # calculate the loss and optimize the network
    loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
    train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

    # initialize the network
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epoch_num):  # epochs loop
            for batch_n in range(1000):  # range(batch_per_ep):  # batches loop
                batch_img, batch_label = mnist.train.next_batch(batch_size)  # read a batch
                batch_img = batch_img.reshape((-1, 28, 28, 1))               # reshape each sample to an (28, 28) image
                batch_img = resize_batch(batch_img)                          # reshape the images to (32, 32)
                _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
                print('Epoch: {} - cost= {:.5f}'.format((ep + 1), c))

        # test the trained network
        batch_img, batch_label = mnist.test.next_batch(50)
        batch_img = resize_batch(batch_img)
        recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]

        # plot the reconstructed images and their ground truths (inputs)
        # fig1 = plt.figure(1)
        # plt.title('Reconstructed Images')
        # print("jjjjjjjjj: " , recon_img.shape)
        # for i in range(50):
        #     plt.subplot(5, 10, i+1)
        #     b = recon_img[i, ..., 0].shape
        #     plt.imshow(recon_img[i, ..., 0], cmap='gray')
        # fig2 = plt.figure(2)
        # plt.title('Input Images')
        # for i in range(50):
        #     plt.subplot(5, 10, i+1)
        #     plt.imshow(batch_img[i, ..., 0], cmap='gray')
        # # plt.show()
        # fig1.savefig('f1.png')
        # fig2.savefig('f2.png')

        # plot the reconstructed images and their ground truths (inputs)
        imgs = []
        imgs_test = []
        titles = []
        for i in range(50):
            imgs.append(batch_img[i, ..., 0])
            imgs_test.append(recon_img[i,...,0])
            titles.append('')
        fig1 = open_figure(1, 'Original Images', (10, 5))
        PlotImages(1, 5, 10, 1, imgs, titles, 'gray', axis=True, colorbar=False)
        fig2 = open_figure(2, 'Test Results', (10, 5))
        PlotImages(2, 5, 10, 1, imgs_test, titles, 'gray', axis=True, colorbar=False)
        fig1.savefig('f1.png')
        fig2.savefig('f2.png')


if __name__ == '__main__':
    train()