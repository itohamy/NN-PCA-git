from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Network import Network
from Plots import open_figure, PlotImages

#batch_size = 64  # Number of samples in each batch
#epoch_num = 1   # Number of epochs to train the network
learn_rate = 0.001        # Learning rate
iterations = 900

def load_data():
    #loading the images
    all_images = np.loadtxt('fashion-mnist_train.csv', delimiter=',',skiprows=1)[:,1:]
    #looking at the shape of the file
    print(all_images.shape)
    # printing something that actually looks like an image
    # plt.imshow(all_images[0].reshape(28,28),cmap='Greys')
    # plt.show()
    all_images = all_images / 255.
    return all_images


def train():

    all_images = load_data()

    device = '/cpu:0'  # '/gpu:0'  OR  '/cpu:0'
    with tf.device(device):
        # build the network
        net = Network()

        input_layer = tf.placeholder('float', [None, 784])
        output_layer = net.autoencoder2(input_layer)

        # output_true shall have the original image for error calculations
        # define our cost function
        meansq = tf.reduce_mean(tf.square(output_layer - input_layer))
        # define our optimizer
        optimizer = tf.train.AdagradOptimizer(learn_rate).minimize(meansq)

        # initialising stuff and starting the session
        init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        batch_size = 100
        epoch_num =10
        tot_images = 60000 # total number of images

        for ep in range(epoch_num):  # epochs loop
            for i in range(int(tot_images/batch_size)):  # batches loop
                # read a batch -> batch_img
                batch_img = all_images[i * batch_size: (i + 1) * batch_size]
                # print("batch size: ", batch_img.shape)
                _, c = sess.run([optimizer,meansq], feed_dict={input_layer: batch_img})
                if not i % 10:
                    print('Epoch {0}: Iteration: {1} Loss: {2:.5f}'.format((ep + 1), i, c))

        # test the trained network, with outliers
        batch_img = all_images[0:batch_size]
        print(batch_img.shape)
        # batch_img[0,...] = generate_outliers(batch_img[0,...],4,10)
        # batch_img[4,...] = generate_outliers(batch_img[4,...],10,15)
        # batch_img[7,...] = generate_outliers(batch_img[7,...],10,25)
        test_results = sess.run([output_layer],feed_dict={input_layer:batch_img})[0]

        # test the trained network

        imgs = []
        imgs_test = []
        titles = []
        for i in range(10):
            I = np.reshape(batch_img[i,...], (28,28))
            imgs.append(I)
            I = np.reshape(test_results[i,...], (28,28))
            imgs_test.append(I)
            titles.append('')
        fig1 = open_figure(1,'Original Images',(7,3))
        PlotImages(1,2,5,1,imgs,titles,'gray',axis=True,colorbar=False)
        fig2 = open_figure(2,'Test Results',(7,3))
        PlotImages(2,2,5,1,imgs_test,titles,'gray',axis=True,colorbar=False)
        fig1.savefig('f1.png')
        fig2.savefig('f2.png')


def generate_outliers(X, s, e):
    X_o = np.reshape(X, X.shape)
    start_idx = np.array([s, s])
    end_idx = np.array([e, e])
    for i in range(start_idx[0], end_idx[0]):
        for j in range(start_idx[1], end_idx[1]):
            X_o[i][j] = 0 #np.random.random_integers(0, 1)
    return X_o


if __name__ == '__main__':
    train()