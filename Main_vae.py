# An undercomplete autoencoder on MNIST dataset
from __future__ import division, print_function, absolute_import
from data_provider import DataProvider
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Plots import open_figure, PlotImages
from vae import VAE


batch_size = 64  # Number of samples in each batch
latent_dim = 6
epoch_num = 1    # Number of epochs to train the network
lr = 0.001        # Learning rate
iterations = 100
img_sz = 128  # This is the panorama size
d_sz = 3 # depth dim of the image

def train():

    # Load data (frames) from video - generate embedded frames of size: (img_sz x img_sz x 4)
    video_name = "movies/BG.mp4"
    data = DataProvider(video_name, img_sz)

    # Learn the subspace using Autoencoder
    device = '/cpu:0' # '/gpu:0'  OR  '/cpu:0'
    with tf.device(device):
        # build the network
        vae = VAE(latent_dim, batch_size, img_sz, d_sz)

    for ep in range(epoch_num):  # epochs loop
        for i in range(iterations):  # batches loop
            # read a batch -> batch_img
            batch_img = data.next_batch(batch_size, 'train')
            loss = vae.update(batch_img)
            if not i % 10:
                print('Epoch {0}: Iteration: {1} Loss: {2:.5f}'.format((ep + 1), i, loss))

    # # test the trained network
    batch_img = data.next_batch(batch_size, 'test')
    batch_img[4, ...] = generate_outliers(batch_img[4,...],50,80)
    batch_img[7, ...] = generate_outliers(batch_img[7,...],40,110)
    test_results = vae.test(batch_img)[0]

    # # plot the reconstructed images and their ground truths (inputs)
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