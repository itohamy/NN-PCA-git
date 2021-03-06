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

    def fusion(self, input_batch):
        new_input = tf.reshape(input_batch, [-1, self.img_sz, self.img_sz, self.d_sz])
        fusionNet = FusionNet()
        out = fusionNet.inference(new_input)
        print("out:", out.shape)
        return tf.reshape(out, [-1, self.img_sz, self.img_sz, self.d_sz])

    def simple1(self, inputs):  # 128x128 only
        # encoder
        # 128 x 128 x 3  ->  64 x 64 x 32  ->  32 x 32 x 16  ->  16 x 16 x 8 ->  8 x 8 x 4 -> 2 x 2 x 2
        net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 8, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 4, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 2, [5, 5], stride=4, padding='SAME')
        # decoder
        # 2 x 2 x 2  ->  8 x 8 x 4  ->  16 x 16 x 8  ->  32 x 32 x 16  ->  64 x 64 x 32  ->  128 x 128 x 3
        net = lays.conv2d_transpose(net, 4, [5, 5], stride=4, padding='SAME')
        net = lays.conv2d_transpose(net, 8, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 16, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, self.d_sz, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
        return net

    def simple2(self, inputs):  # 256x256 only
        # encoder
        # 256x256x3 -> 128x128x64 -> 64x64x32 -> 32x32x16 -> 16x16x8 -> 8x8x4 -> 2x2x2
        net = lays.conv2d(inputs, 64, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 32, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 8, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 4, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d(net, 2, [5, 5], stride=4, padding='SAME')
        # decoder
        # 2x2x2 -> 8x8x4 -> 16x16x8 -> 32x32x16 -> 64x64x32 -> 128x128x64 -> 256x256x3
        net = lays.conv2d_transpose(net, 4, [5, 5], stride=4, padding='SAME')
        net = lays.conv2d_transpose(net, 8, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 16, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, 64, [5, 5], stride=2, padding='SAME')
        net = lays.conv2d_transpose(net, self.d_sz, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
        return net

    def vae(self, input_layer):
        return

    def fully_connected(self, input_layer): # 128x128 only
        # encoder
        n_nodes_inpl = 784
        n_nodes_hl1 = 32
        # decoder
        n_nodes_hl2 = 32
        n_nodes_outl = 784

        # first hidden layer has 784*32 weights and 32 biases
        hidden_1_layer_vals = {
            'weights':tf.Variable(tf.random_normal([n_nodes_inpl,n_nodes_hl1])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
        # second hidden layer has 32*32 weights and 32 biases
        hidden_2_layer_vals = {
            'weights':tf.Variable(tf.random_normal([n_nodes_hl1,n_nodes_hl2])),
            'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}
        # second hidden layer has 32*784 weights and 784 biases
        output_layer_vals = {
            'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_outl])),
            'biases':tf.Variable(tf.random_normal([n_nodes_outl]))}

        # image with shape 784 goes in

        # multiply output of input_layer wth a weight matrix and add biases
        layer_1 = tf.nn.sigmoid(
            tf.add(tf.matmul(input_layer,hidden_1_layer_vals['weights']),hidden_1_layer_vals['biases']))
        # multiply output of layer_1 wth a weight matrix and add biases
        layer_2 = tf.nn.sigmoid(
            tf.add(tf.matmul(layer_1,hidden_2_layer_vals['weights']),hidden_2_layer_vals['biases']))
        # multiply output of layer_2 wth a weight matrix and add biases
        output_layer = tf.matmul(layer_2,output_layer_vals['weights']) + output_layer_vals['biases']

        return output_layer


class FusionNet(object):
    def __init__(self):
        self.act_fn = acts.pRelu
        self.kernel_num = 32
        self.output_dim = 3
        self.log = 1
        print("FusionNet Loading"),

    def skip_connection(self, input_, output_):
        return tf.add(input_, output_)

    def res_block_with_n_conv_layers(self, input_, output_dim, num_repeat, name="res_block"):
        output_ = layers.conv2d_same_repeat(input_, output_dim,
                                            num_repeat=num_repeat, activation_fn=self.act_fn, name=name)
        return self.skip_connection(input_, output_)

    def res_block_with_3_conv_layers(self, input_, output_dim, name="res_block"):
        return self.res_block_with_n_conv_layers(input_, output_dim, num_repeat=3, name=name)

    def conv_res_conv_block(self, input_, output_dim, name="conv_res_conv_block"):
        with tf.variable_scope(name):
            conv1 = layers.conv2d_same_act(input_, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv1")
            res = self.res_block_with_3_conv_layers(conv1, output_dim, name="res_block")
            conv2 = layers.conv2d_same_act(res, output_dim, activation_fn=self.act_fn,
                                           with_logit=False, name="conv2")
            return conv2

    def encoder(self, input_):
        self.down1 = self.conv_res_conv_block(input_, 32, name="down1")
        pool1 = layers.max_pool(self.down1, name="pool1")

        self.down2 = self.conv_res_conv_block(pool1, 16, name="down2")
        pool2 = layers.max_pool(self.down2, name="pool2")

        self.down3 = self.conv_res_conv_block(pool2, 8, name="down3")
        pool3 = layers.max_pool(self.down3, name="pool3")

        self.down4 = self.conv_res_conv_block(pool3, 4, name="down4")
        pool4 = layers.max_pool(self.down4, name="pool4")

        self.down5 = self.conv_res_conv_block(pool4, 2, name="down5")
        pool5 = layers.max_pool(self.down5, name="pool5")

        self.down6 = self.conv_res_conv_block(pool5, 2, name="down6")
        pool6 = layers.max_pool(self.down6, name="pool6")


        if self.log == 1:
            print("encoder input : ", input_.get_shape())
            print("conv1 : ", self.down1.get_shape())
            print("pool1 : ", pool1.get_shape())
            print("conv2 : ", self.down2.get_shape())
            print("pool2 : ", pool2.get_shape())
            print("conv3 : ", self.down3.get_shape())
            print("pool3 : ", pool3.get_shape())
            print("conv4 : ", self.down4.get_shape())
            print("pool4 : ", pool4.get_shape())
            print("conv5 : ", self.down5.get_shape())
            print("pool5 : ", pool5.get_shape())
            print("conv6 : ", self.down6.get_shape())
            print("pool6 : ", pool6.get_shape())

        return pool6

    def decoder(self, input_):
        conv_trans6 = layers.conv2dTrans_same_act(input_,self.down6.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool6")
        res6 = self.skip_connection(conv_trans6, self.down6)
        up6 = self.conv_res_conv_block(res6, 2, name="up6")

        conv_trans5 = layers.conv2dTrans_same_act(up6, self.down5.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool5")
        res5 = self.skip_connection(conv_trans5, self.down5)
        up5 = self.conv_res_conv_block(res5, 2, name="up5")

        conv_trans4 = layers.conv2dTrans_same_act(up5, self.down4.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool4")
        res4 = self.skip_connection(conv_trans4, self.down4)
        up4 = self.conv_res_conv_block(res4, 4, name="up4")

        conv_trans3 = layers.conv2dTrans_same_act(up4, self.down3.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool3")
        res3 = self.skip_connection(conv_trans3, self.down3)
        up3 = self.conv_res_conv_block(res3, 8, name="up3")

        conv_trans2 = layers.conv2dTrans_same_act(up3, self.down2.get_shape(),
                                                  activation_fn=self.act_fn, with_logit=False, name="unpool2")
        res2 = self.skip_connection(conv_trans2, self.down2)
        up2 = self.conv_res_conv_block(res2, 16, name="up2")

        conv_trans1 = layers.conv2dTrans_same_act(up2, self.down1.get_shape(),
                                                 activation_fn=self.act_fn, with_logit=False, name="unpool1")
        res1 = self.skip_connection(conv_trans1, self.down1)
        up1 = self.conv_res_conv_block(res1, 32, name="up1")

        if self.log == 1:
            print("dncoder input : ",input_.get_shape())
            print("convT1 : ", conv_trans6.get_shape())
            print("res1 : ", res6.get_shape())
            print("up1 : ", up6.get_shape())
            print("convT2 : ", conv_trans5.get_shape())
            print("res2 : ", res5.get_shape())
            print("up2 : ", up5.get_shape())
            print("convT3 : ", conv_trans4.get_shape())
            print("res3 : ", res4.get_shape())
            print("up3 : ", up4.get_shape())
            print("convT4 : ", conv_trans3.get_shape())
            print("res4 : ", res3.get_shape())
            print("up4 : ", up3.get_shape())
            print("convT5 : ", conv_trans2.get_shape())
            print("res5 : ", res2.get_shape())
            print("up5 : ", up2.get_shape())
            print("convT6 : ", conv_trans1.get_shape())
            print("res6 : ", res1.get_shape())
            print("up6 : ", up1.get_shape())

        return up1

    def inference(self, input_):
        encode_vec = self.encoder(input_)
        bridge = self.conv_res_conv_block(encode_vec, 2, name="bridge")
        decode_vec = self.decoder(bridge)
        output = layers.bottleneck_layer(decode_vec, self.output_dim, name="output")

        if self.log == 1:
            print("output : ", output.get_shape())

        print("Completed!!")

        return output
