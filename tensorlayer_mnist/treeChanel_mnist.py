#! /usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import set_keep
import time

def tryInput():
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    y_test = np.asarray(y_test, dtype=np.int32)
    print(y_test[1:])

def transformImage(images, labels):
    colored_images = np.zeros((images.shape[0],images.shape[1], images.shape[2], 3), dtype=np.float32)
    color = np.zeros((labels.size,), dtype=np.int32)
    for i in range(images.shape[0]):
        index = labels[i] // 3
        if index == 3: index = 2
        color[i] = index
        for j in range(images.shape[1]):
            for k in range(images.shape[2]):
                colored_images[i][j][k][index] = images[i][j][k]
    return [colored_images, color]

def main_test_cnn_layer():
    X_train, y_train, X_val, y_val, X_test, y_test = \
        tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))

    y_1_train = np.asarray(y_train, dtype=np.int32)
    X_train, y_2_train = transformImage(np.asarray(X_train, dtype=np.float32), y_1_train)
    y_1_val = np.asarray(y_val, dtype=np.int32)
    X_val, y_1_val = transformImage(np.asarray(X_val, dtype=np.float32), y_1_val)
    y_1_test = np.asarray(y_test, dtype=np.int32)
    X_test, y_2_test = transformImage(np.asarray(X_test, dtype=np.float32), y_1_test)

    print('X_train.shape', X_train.shape)
    print('y_train.shape', y_train.shape)
    print('X_val.shape', X_val.shape)
    print('y_val.shape', y_val.shape)
    print('X_test.shape', X_test.shape)
    print('y_test.shape', y_test.shape)
    print('X %s   y %s' % (X_test.dtype, y_test.dtype))

    sess = tf.InteractiveSession()

    # Define the batchsize at the begin, you can give the batchsize in x and y_
    # rather than 'None', this can allow TensorFlow to apply some optimizations
    # – especially for convolutional layers.
    batch_size = 128

    x = tf.placeholder(tf.float32, shape=[batch_size, 28, 28, 3])  # [batch_size, height, width, channels]
    y_ = tf.placeholder(tf.int64, shape=[batch_size, ])

    network = tl.layers.InputLayer(x, name='input_layer')
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[5, 5, 3, 32],  # 32 features for each 5x5 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='cnn_layer1')  # output: (?, 28, 28, 32)
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer1', )  # output: (?, 14, 14, 32)
    
    network = tl.layers.Conv2dLayer(network,
                                    act=tf.nn.relu,
                                    shape=[5, 5, 32, 64],  # 64 features for each 5x5 patch
                                    strides=[1, 1, 1, 1],
                                    padding='SAME',
                                    name='cnn_layer2')  # output: (?, 14, 14, 64)
    network = tl.layers.PoolLayer(network,
                                  ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1],
                                  padding='SAME',
                                  pool=tf.nn.max_pool,
                                  name='pool_layer2', )  # output: (?, 7, 7, 64)
    network = tl.layers.FlattenLayer(network, name='flatten_layer')  # output: (?, 3136)
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop1')  # output: (?, 3136)
    network = tl.layers.DenseLayer(network, n_units=256,
                                   act=tf.nn.relu, name='relu1')  # output: (?, 256)
    network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')  # output: (?, 256)
    network = tl.layers.DenseLayer(network, n_units=10,
                                   act=tf.identity,
                                   name='output_layer')  # output: (?, 10)

    y = network.outputs

    ce = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y, y_))
    cost = ce

    correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # train
    n_epoch = 200
    learning_rate = 0.0001
    print_freq = 10

    train_params = network.all_params
    train_op = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999,
                                      epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

    sess.run(tf.initialize_all_variables())
    network.print_params()
    network.print_layers()

    print('   learning_rate: %f' % learning_rate)
    print('   batch_size: %d' % batch_size)

    for epoch in range(n_epoch):
        start_time = time.time()
        print(str(epoch) + "startime: " + str(start_time)),
        for X_train_a, y_train_a in tl.iterate.minibatches(
            X_train, y_train, batch_size, shuffle=True):
            feed_dict = {x: X_train_a, y_: y_train_a}
            feed_dict.update(network.all_drop)  # enable noise layers
            sess.run(train_op, feed_dict=feed_dict)

        if epoch + 1 == 1 or (epoch + 1) % print_freq == 0:
            print("Epoch %d of %d took %fs" % (epoch + 1, n_epoch, time.time() - start_time))
            train_loss, train_acc, n_batch = 0, 0, 0
            for X_train_a, y_train_a in tl.iterate.minibatches(
                X_train, y_train, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                feed_dict = {x: X_train_a, y_: y_train_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                train_loss += err;
                train_acc += ac;
                n_batch += 1
            print("   train loss: %f" % (train_loss / n_batch))
            print("   train acc: %f" % (train_acc / n_batch))
            val_loss, val_acc, n_batch = 0, 0, 0
            for X_val_a, y_val_a in tl.iterate.minibatches(
                X_val, y_val, batch_size, shuffle=True):
                dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
                feed_dict = {x: X_val_a, y_: y_val_a}
                feed_dict.update(dp_dict)
                err, ac = sess.run([cost, acc], feed_dict=feed_dict)
                val_loss += err;
                val_acc += ac;
                n_batch += 1
            print("   val loss: %f" % (val_loss / n_batch))
            print("   val acc: %f" % (val_acc / n_batch))
            # try:
            #     # tl.visualize.CNN2d(network.all_params[0].eval(),
            #     #                    second=10, saveable=True,
            #     #                    name='cnn1_' + str(epoch + 1), fig_idx=2012)
            # except:
            #     raise Exception(
            #         "# You should change visualize.CNN(), if you want to save the feature images for different dataset")

    print('Evaluation')
    test_loss, test_acc, n_batch = 0, 0, 0
    for X_test_a, y_test_a in tl.iterate.minibatches(
        X_test, y_test, batch_size, shuffle=True):
        dp_dict = tl.utils.dict_to_one(network.all_drop)  # disable noise layers
        feed_dict = {x: X_test_a, y_: y_test_a}
        feed_dict.update(dp_dict)
        err, ac = sess.run([cost, acc], feed_dict=feed_dict)
        test_loss += err;
        test_acc += ac;
        n_batch += 1
    print("   test loss: %f" % (test_loss / n_batch))
    print("   test acc: %f" % (test_acc / n_batch))

if __name__ == '__main__':

    sess = tf.InteractiveSession()
    sess = tl.ops.set_gpu_fraction(sess, gpu_fraction = 0.3)
    try:
        tryInput()
        # main_test_cnn_layer()
        tl.ops.exit_tf(sess)  # close sess, tensorboard and nvidia-process
    except KeyboardInterrupt:
        print('\nKeyboardInterrupt')
        tl.ops.exit_tf(sess)
