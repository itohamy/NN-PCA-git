from tensorflow.contrib import layers
import tensorflow as tf
import layers
import acts
import tensorflow.contrib.layers as lays
BN_EPSILON = 0.001


class Network():

    training = tf.placeholder(tf.bool)

    def __init__(self, img_sz, d_sz):
        self.img_sz = img_sz
        self.d_sz = d_sz
        pass

    def encoder(self, x, keep_prob, latent_dim):  # 128x128 only
        # encoder
        # 128 x 128 x 3  ->  64 x 64 x 16  ->  32 x 32 x 32  ->  16 x 16 x 64 ->  8 x 8 x 128 -> flat
        net = lays.conv2d(x, 32, [5, 5], stride=2, padding='SAME')
        net = tf.nn.dropout(net, keep_prob)
        net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
        net = tf.nn.dropout(net, keep_prob)
        net = lays.conv2d(net, 8, [5, 5], stride=2, padding='SAME')
        net = tf.nn.dropout(net, keep_prob)
        net = lays.conv2d(net, 4, [5, 5], stride=2, padding='SAME')
        net = layers.flatten(net)
        net = lays.fully_connected(net, 500, scope='fc-01')
        net = lays.fully_connected(net, 200, scope='fc-02')
        net = lays.fully_connected(net, 2 * latent_dim, activation_fn=None, scope='fc-final')
        return net

    def decoder(self, z, keep_prob):  # gets z which is in a shape of array of size "latent_dim"
        # decoder
        # 8 x 8 x 4  ->  16 x 16 x 8  ->  32 x 32 x 16  ->  64 x 64 x 32  ->  128 x 128 x 3
        dropout_prob = tf.placeholder(tf.float32,name='dropout_probability')
        x = tf.expand_dims(z, 1)
        x = tf.expand_dims(x, 1)
        x = lays.conv2d_transpose(x, 4, [5, 5], stride=4, padding='SAME')
        x = tf.nn.dropout(x, keep_prob)
        x = lays.conv2d_transpose(x, 8, [5, 5], stride=4, padding='SAME')
        x = tf.nn.dropout(x, keep_prob)
        x = lays.conv2d_transpose(x, 16, [5, 5], stride=2, padding='SAME')
        x = tf.nn.dropout(x, keep_prob)
        x = lays.conv2d_transpose(x, 32, [5, 5], stride=2, padding='SAME')
        x = lays.conv2d_transpose(x, self.d_sz, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh) # activation_fn=tf.nn.tanh
        #x = layers.flatten(x)
        return x

