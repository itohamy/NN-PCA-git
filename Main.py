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
iterations = 50
img_sz = 128  # This is the panorama size
d_sz = 3


def train():

    # Load data (frames) from video - generate embedded frames of size: (img_sz x img_sz x 4)
    video_name = "movies/BG.mp4"
    data = DataProvider(video_name, img_sz)

    # STN - align the frames to the right position in the panorama
    # align_frames(data)

    # Learn the subspace using Autoencoder
    device = '/cpu:0' # '/gpu:0'  OR  '/cpu:0'
    with tf.device(device):
        # build the network
        net = Network(img_sz, d_sz)

        # calculate the number of batches per epoch
        # batch_per_ep = data.train_size // batch_size
        # print('Data size: ', data.train_size, ' Num of epochs: ', epoch_num, ' Batches per epoch: ', batch_per_ep)

        ae_inputs = tf.placeholder(tf.float32, (batch_size, img_sz, img_sz, 3))  # input to the network
        ae_outputs = net.simple1(ae_inputs) # fusion , simple1 , simple2, vae, fully_connected

        # calculate the loss and optimize the network
        loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))  # claculate the mean square error loss
        train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)

        # initialize the network
        init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for ep in range(epoch_num):  # epochs loop
            for i in range(iterations):  # batches loop
                # read a batch -> batch_img
                batch_img = data.next_batch(batch_size, 'train')
                # print("batch size: ", batch_img.shape)
                _, c = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
                if not i % 10:
                    print('Epoch {0}: Iteration: {1} Loss: {2:.5f}'.format((ep + 1), i, c))

        # test the trained network
        batch_img = data.next_batch(batch_size, 'train')
        #batch_img[4, ...] = generate_outliers(batch_img[4,...],50,80)
        # batch_img[7, ...] = generate_outliers(batch_img[7,...],40,110)
        test_results = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]

        # plot the reconstructed images and their ground truths (inputs)
        imgs = []
        imgs_test = []
        titles = []

        for i in range(10):
            imgs.append(batch_img[i, ...])
            imgs_test.append(np.abs(test_results[i, ...]))
            titles.append('')
        fig1 = open_figure(1, 'Original Images', (7, 3))
        PlotImages(1, 2, 5, 1, imgs, titles, 'gray', axis=True, colorbar=False)
        fig2 = open_figure(2, 'Test Results', (7, 3))
        PlotImages(2, 2, 5, 1, imgs_test, titles, 'gray', axis=True, colorbar=False)
        plt.show()
        fig1.savefig('f1.png')
        fig2.savefig('f2.png')


# Use STN
def align_frames(I):
    return


# Re-transform the image to the original position
# Remove W and return an image with normal size
def post_process(data):
    return


def generate_outliers(X, s, e):
    X_o = np.reshape(X, X.shape)
    start_idx = np.array([s, s])
    end_idx = np.array([e, e])
    for i in range(start_idx[0], end_idx[0]):
        for j in range(start_idx[1], end_idx[1]):
            X_o[i][j] = np.random.random_integers(0, 1)
    return X_o


if __name__ == '__main__':
    train()