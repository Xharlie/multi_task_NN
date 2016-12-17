# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mnist_util import *


import argparse
import gzip
import os
import sys
import time

import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

from parameter import *


def variable_summaries(var, name):
  """Attach a lot of summaries to a Tensor."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.scalar_summary('mean/' + name, mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.scalar_summary('stddev/' + name, stddev)
    tf.scalar_summary('max/' + name, tf.reduce_max(var))
    tf.scalar_summary('min/' + name, tf.reduce_min(var))
    tf.histogram_summary(name, var)

def inference():

  # The variables below hold all the trainable weights. They are passed an
  # initial value which will be assigned when we call:
  # {tf.initialize_all_variables().run()}
  conv1_weights = tf.Variable(
      tf.truncated_normal([5, 5, NUM_CHANNELS, 32],  # 5x5 filter, depth 32.
                          stddev=PARA_INIT_STD,
                          seed=SEED, dtype=data_type(FLAGS)))
  conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type(FLAGS)))

  fcclr_weights = tf.Variable(  # fully connected, depth 512.
    tf.truncated_normal([IMAGE_SIZE // 2 * IMAGE_SIZE //2 * 32, NUM_COLORS],
                        stddev=PARA_INIT_STD,
                        seed=SEED,
                        dtype=data_type(FLAGS)))
  fcclr_biases = tf.Variable(tf.constant(PARA_INIT_CONS, shape=[NUM_COLORS], dtype=data_type(FLAGS)))

  conv3_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 32, 64], stddev=PARA_INIT_STD,
      seed=SEED, dtype=data_type(FLAGS)))
  conv3_biases = tf.Variable(tf.constant(PARA_INIT_CONS, shape=[64], dtype=data_type(FLAGS)))

  fccharset_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, NUM_CHARSET],
                          stddev=PARA_INIT_STD,
                          seed=SEED,
                          dtype=data_type(FLAGS)))
  fccharset_biases = tf.Variable(tf.constant(PARA_INIT_CONS, shape=[NUM_CHARSET], dtype=data_type(FLAGS)))

  conv5_weights = tf.Variable(tf.truncated_normal(
      [5, 5, 64, 64], stddev=PARA_INIT_STD,
      seed=SEED, dtype=data_type(FLAGS)))
  conv5_biases = tf.Variable(tf.constant(PARA_INIT_CONS, shape=[64], dtype=data_type(FLAGS)))

  fc1_weights = tf.Variable(  # fully connected, depth 512.
      tf.truncated_normal([IMAGE_SIZE // 8 * IMAGE_SIZE // 8 * 64, 512],
                          stddev=PARA_INIT_STD,
                          seed=SEED,
                          dtype=data_type(FLAGS)))
  fc1_biases = tf.Variable(tf.constant(PARA_INIT_CONS, shape=[512], dtype=data_type(FLAGS)))
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=PARA_INIT_STD,
                                                seed=SEED,
                                                dtype=data_type(FLAGS)))
  fc2_biases = tf.Variable(tf.constant(
    PARA_INIT_CONS, shape=[NUM_LABELS], dtype=data_type(FLAGS)))

  variable_summaries(conv1_weights, "conv1_weights") #!!!
  variable_summaries(conv1_biases, "conv1_biases")  #!!!
  variable_summaries(fcclr_weights, "fcclr_weights") #!!!
  variable_summaries(fcclr_biases, "fcclr_biases")  #!!!
  variable_summaries(conv3_weights, "conv3_weights") #!!!
  variable_summaries(conv3_biases, "conv3_biases")  #!!!
  variable_summaries(fccharset_weights, "fccharset_weights") #!!!
  variable_summaries(fccharset_biases, "fccharset_biases")  #!!!
  variable_summaries(conv5_weights, "conv5_weights") #!!!
  variable_summaries(conv5_biases, "conv5_biases")  #!!!
  variable_summaries(fc1_weights, "fc1_weights") #!!!
  variable_summaries(fc1_biases, "fc1_biases")  #!!!
  variable_summaries(fc2_weights, "fc2_weights") #!!!
  variable_summaries(fc2_biases, "fc2_biases")  #!!!

  return conv1_weights, conv1_biases, \
          fcclr_weights, fcclr_biases, \
          conv3_weights, conv3_biases, \
          fccharset_weights, fccharset_biases, \
          conv5_weights, conv5_biases, \
          fc1_weights, fc1_biases, \
          fc2_weights, fc2_biases

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, conv1_weights, conv1_biases,
          fcclr_weights, fcclr_biases,
          conv3_weights, conv3_biases,
          fccharset_weights, fccharset_biases,
          conv5_weights, conv5_biases,
          fc1_weights, fc1_biases,
          fc2_weights, fc2_biases, train=False,):
  """The Model definition."""
  # 2D convolution, with 'SAME' padding (i.e. the output feature map has
  # the same size as the input). Note that {strides} is a 4D array whose
  # shape matches the data layout: [image index, y, x, depth].
  with tf.name_scope('conv1') as scope:
    conv = tf.nn.conv2d(data,
                        conv1_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    # Bias and rectified linear non-linearity.
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv1_biases), name=scope)
    # Max pooling. The kernel size spec {ksize} also follows the layout of
    # the data. Here we have a pooling window of 2, and a stride of 2.
    if train: tf.histogram_summary(relu.op.name + '/activations', relu) #!!!
    print_activations(relu)

  with tf.name_scope('pool2') as scope:
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME', name=scope)
    print_activations(pool)
    pool_shape = pool.get_shape().as_list()

  with tf.name_scope('fcclr') as scope:
    reshapeclr = tf.reshape(
      pool,
      [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    print_activations(reshapeclr)

  with tf.name_scope('conv3') as scope:

    conv = tf.nn.conv2d(pool,
                        conv3_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv3_biases), name=scope)
    print_activations(relu)
    if train: tf.histogram_summary(relu.op.name + '/activations', relu) #!!!!

  with tf.name_scope('pool4') as scope:
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',name=scope)
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    print_activations(pool)

  with tf.name_scope('fccharset') as scope:
    reshapecharset = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    print_activations(reshapecharset)

  with tf.name_scope('conv5') as scope:

    conv = tf.nn.conv2d(pool,
                        conv5_weights,
                        strides=[1, 1, 1, 1],
                        padding='SAME')
    relu = tf.nn.relu(tf.nn.bias_add(conv, conv5_biases),name=scope)
    print_activations(relu)
    if train: tf.histogram_summary(relu.op.name + '/activations', relu) #!!!!

  with tf.name_scope('pool6') as scope:
    pool = tf.nn.max_pool(relu,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME',name=scope)
    # Reshape the feature map cuboid into a 2D matrix to feed it to the
    # fully connected layers.
    pool_shape = pool.get_shape().as_list()
    print_activations(pool)

  with tf.name_scope('fc1') as scope:
    reshape = tf.reshape(
        pool,
        [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
    # Fully connected layer. Note that the '+' operation automatically
    # broadcasts the biases.
    hidden = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases,name=scope)
    print_activations(hidden)

  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  return tf.matmul(hidden, fc2_weights) + fc2_biases, \
         tf.matmul(reshapeclr, fcclr_weights) + fcclr_biases, \
         tf.matmul(reshapecharset, fccharset_weights) + fccharset_biases, \
         {"fc1_weights":fc1_weights, "fc1_biases":fc1_biases,
          "fc2_weights":fc2_weights, "fc2_biases":fc2_biases,
          "fccharset_weights":fccharset_weights, "fccharset_biases":fccharset_biases,
          "fcclr_weights":fcclr_weights, "fcclr_biases":fcclr_biases}

# Small utility function to evaluate a dataset by feeding batches of data to
# {eval_data} and pulling the results from {eval_predictions}.
# Saves memory and enables this to run on smaller GPUs.
def eval_in_batches(data, sess, eval_label_prediction, eval_color_prediction, eval_charset_prediction, eval_data_node):
  """Get all predictions for a dataset by running it in small batches."""
  size = data.shape[0]
  if size < EVAL_BATCH_SIZE:
    raise ValueError("batch size for evals larger than dataset: %d" % size)
  predictions_label_result = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
  predictions_color_result = numpy.ndarray(shape=(size, NUM_COLORS), dtype=numpy.float32)
  predictions_charset_result = numpy.ndarray(shape=(size, NUM_CHARSET), dtype=numpy.float32)
  for begin in xrange(0, size, EVAL_BATCH_SIZE):
    end = begin + EVAL_BATCH_SIZE
    if end <= size:
      predictions_label_result[begin:end, :], predictions_color_result[begin:end, :] \
        , predictions_charset_result[begin:end, :] = sess.run(
        [eval_label_prediction, eval_color_prediction, eval_charset_prediction],
        feed_dict={eval_data_node: data[begin:end, ...]})
    else:
      batch_label_predictions, batch_color_predictions, batch_charset_predictions = sess.run(
        [eval_label_prediction, eval_color_prediction, eval_charset_prediction],
        feed_dict={eval_data_node: data[-EVAL_BATCH_SIZE:, ...]})
      predictions_label_result[begin:, :] = batch_label_predictions[begin - size:, :]
      predictions_color_result[begin:, :] = batch_color_predictions[begin - size:, :]
      predictions_charset_result[begin:, :] = batch_charset_predictions[begin - size:, :]
  return predictions_label_result, predictions_color_result, predictions_charset_result

def load():
  if FLAGS.self_test:
    print('Running self-test.')
    train_data, train_labels = fake_data(256)
    validation_data, validation_labels = fake_data(EVAL_BATCH_SIZE)
    test_data, test_labels = fake_data(EVAL_BATCH_SIZE)
    num_epochs = 1
  else:
    # Get the data.
    data_filename = WORK_DIRECTORY + '/image_source.dat'
    label_filename = WORK_DIRECTORY + '/label_source.dat'
    color_filename = WORK_DIRECTORY + '/color_source.dat'
    charSet_filename = WORK_DIRECTORY + '/charSet_source.dat'

    # Extract it into numpy arrays.
    labels = extract_labels(label_filename, NUM_SAMPLES)
    charSet = extract_labels(charSet_filename, NUM_SAMPLES)
    color = extract_labels(color_filename, NUM_SAMPLES)
    data = extract_data(data_filename, NUM_SAMPLES)

    # Generate a validation set.
    print(TRAINING_SIZE)
    train_data = data[:TRAINING_SIZE , ...]
    train_labels = labels[:TRAINING_SIZE]
    train_colors = color[:TRAINING_SIZE]
    train_charSet = charSet[:TRAINING_SIZE]

    print(VALIDATION_SIZE)
    validation_data = data[TRAINING_SIZE: TRAINING_SIZE + VALIDATION_SIZE, ...]
    validation_colors = color[TRAINING_SIZE: TRAINING_SIZE + VALIDATION_SIZE]
    validation_labels = labels[TRAINING_SIZE: TRAINING_SIZE + VALIDATION_SIZE]
    validation_charSet = charSet[TRAINING_SIZE: TRAINING_SIZE + VALIDATION_SIZE]

    test_data = data[TRAINING_SIZE + VALIDATION_SIZE:, ...]
    test_colors = color[TRAINING_SIZE + VALIDATION_SIZE:]
    test_labels = labels[TRAINING_SIZE + VALIDATION_SIZE:]
    test_charSet = charSet[TRAINING_SIZE + VALIDATION_SIZE:]

  return train_data, train_labels, train_colors, train_charSet, \
         validation_data, validation_labels, validation_colors, validation_charSet, \
         test_data, test_labels, test_colors, test_charSet

def main(argv=None):  # pylint: disable=unused-argument
  if not tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

  with tf.Session() as sess:
    train_data, train_labels, train_colors, train_charSet, \
    validation_data, validation_labels, validation_colors, validation_charSet, \
    test_data, test_labels, test_colors, test_charSet = load()

    num_epochs = NUM_EPOCHS
    train_size = train_labels.shape[0]
    # This is where training samples and labels are fed to the graph.
    # These placeholder nodes will be fed a batch of training data at each
    # training step using the {feed_dict} argument to the Run() call below.
    train_data_node = tf.placeholder(
        data_type(FLAGS),
        shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    train_colors_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    train_charSet_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,))
    eval_data_node = tf.placeholder(
        data_type(FLAGS),
        shape=(EVAL_BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))

    tf.image_summary('input', train_data_node, 10)  #!!!

    # Training computation: logits + cross-entropy loss.
    conv1_weights, conv1_biases, \
    fcclr_weights, fcclr_biases, \
    conv3_weights, conv3_biases, \
    fccharset_weights, fccharset_biases, \
    conv5_weights, conv5_biases, \
    fc1_weights, fc1_biases, \
    fc2_weights, fc2_biases = inference()

    logits, logitsclr, logitscharset, regul = model(train_data_node,conv1_weights, conv1_biases, \
                                      fcclr_weights, fcclr_biases, \
                                      conv3_weights, conv3_biases, \
                                      fccharset_weights, fccharset_biases, \
                                      conv5_weights, conv5_biases, \
                                      fc1_weights, fc1_biases, \
                                      fc2_weights, fc2_biases, True)

    evallogits, evallogitsclr, evallogitscharset, _ = model(eval_data_node, conv1_weights, conv1_biases, \
                                      fcclr_weights, fcclr_biases, \
                                      conv3_weights, conv3_biases, \
                                      fccharset_weights, fccharset_biases, \
                                      conv5_weights, conv5_biases, \
                                      fc1_weights, fc1_biases, \
                                      fc2_weights, fc2_biases, False)

    label_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, train_labels_node))
    color_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logitsclr, train_colors_node))
    charset_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logitscharset, train_charSet_node))
    loss = LABEL_LOSS_WEIGHT * label_loss + COLOR_LOSS_WEIGHT * color_loss + CHARSET_LOSS_WEIGHT * charset_loss
    tf.scalar_summary('label_loss', label_loss) #!!!
    tf.scalar_summary('color_loss', color_loss) #!!!
    tf.scalar_summary('charset_loss', charset_loss) #!!!
    tf.scalar_summary('weighted_loss', loss)  #!!!

    # L2 regularization for the fully connected parameters.
    regularizers = REGULRATE * (
                    LABEL_LOSS_WEIGHT * (tf.nn.l2_loss(regul["fc1_weights"]) + tf.nn.l2_loss(regul["fc1_biases"]) +
                    tf.nn.l2_loss(regul["fc2_weights"]) + tf.nn.l2_loss(regul["fc2_biases"])) \
                    + COLOR_LOSS_WEIGHT * (tf.nn.l2_loss(regul["fcclr_weights"]) + tf.nn.l2_loss(regul["fcclr_biases"]))
                    + CHARSET_LOSS_WEIGHT * (tf.nn.l2_loss(regul["fccharset_weights"]) + tf.nn.l2_loss(regul["fccharset_biases"]))
    )
    tf.scalar_summary('regularizers', regularizers) #!!!
    # Add the regularization term to the loss.
    loss += regularizers
    tf.scalar_summary('loss', loss) #!!!

    # Optimizer: set up a variable that's incremented once per batch and
    # controls the learning rate decay.
    batch = tf.Variable(0, dtype=data_type(FLAGS))
    # Decay once per epoch, using an exponential schedule starting at 0.01.
    learning_rate = tf.train.exponential_decay(
        BASE_LRN_RATE,       # Base learning rate.
        batch * BATCH_SIZE,  # Current index into the dataset.
        train_size,          # Decay step.
        0.95,                # Decay rate.
        staircase=True)
    # Use simple momentum for the optimization.
    tf.scalar_summary('learning_rate', learning_rate) #!!!
    optimizer = tf.train.MomentumOptimizer(learning_rate,
                                           0.9).minimize(loss,
                                                         global_step=batch)

    # Predictions for the current training minibatch.
    train_label_prediction = tf.nn.softmax(logits)
    train_color_prediction = tf.nn.softmax(logitsclr)
    train_charset_prediction = tf.nn.softmax(logitscharset)

    # Predictions for the test and validation, which we'll compute less often.
    eval_label_prediction = tf.nn.softmax(evallogits)
    eval_color_prediction = tf.nn.softmax(evallogitsclr)
    eval_charset_prediction = tf.nn.softmax(evallogitscharset)

    with tf.name_scope('accuracy'):
      with tf.name_scope('error_prediction'):
        correct_train_label_prediction = tf.equal(tf.argmax(train_label_prediction, 1), train_labels_node)
        correct_train_color_prediction = tf.equal(tf.argmax(train_color_prediction, 1), train_colors_node)
        correct_train_charset_prediction = tf.equal(tf.argmax(train_charset_prediction, 1), train_charSet_node)
      with tf.name_scope('error'):
        label_error = 1 - tf.reduce_mean(tf.cast(correct_train_label_prediction, tf.float32))
        color_error = 1 - tf.reduce_mean(tf.cast(correct_train_color_prediction, tf.float32))
        charset_error = 1 - tf.reduce_mean(tf.cast(correct_train_charset_prediction, tf.float32))
      tf.scalar_summary('label_error', label_error) #!!!
      tf.scalar_summary('color_error', color_error) #!!!
      tf.scalar_summary('charset_error', charset_error) #!!!

    merged = tf.merge_all_summaries()
    train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/' + str(REGULRATE) + '/' + str(COLOR_LOSS_WEIGHT) + '/train',
                                          sess.graph)

    # Create a local session to run the training.
    start_time = time.time()
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(num_epochs * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      batch_colors = train_colors[offset:(offset + BATCH_SIZE)]
      batch_charset = train_charSet[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels,
                   train_colors_node: batch_colors,
                   train_charSet_node: batch_charset}
      # Run the graph and fetch some of the nodes.
      _, l, lr, label_predictions, color_predictions, charset_predictions = sess.run(
          [optimizer, loss, learning_rate, train_label_prediction, train_color_prediction, train_charset_prediction],
          feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        train_summary = sess.run(merged, feed_dict=feed_dict)
        train_writer.add_summary(train_summary, step)
        train_writer.flush()
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch label error: %.1f%%' % error_rate(label_predictions, batch_labels))
        print('Minibatch color error: %.1f%%' % error_rate(color_predictions, batch_colors))
        print('Minibatch charset error: %.1f%%' % error_rate(charset_predictions, batch_charset))
        eval_label_predictions, eval_color_predictions, eval_charset_predictions = eval_in_batches(validation_data, sess,
                               eval_label_prediction, eval_color_prediction, eval_charset_prediction, eval_data_node)
        print('Validation label error: %.1f%%' % error_rate(eval_label_predictions, validation_labels))
        print('Validation color error: %.1f%%' % error_rate(eval_color_predictions, validation_colors))
        print('Validation charset error: %.1f%%' % error_rate(eval_charset_predictions, validation_charSet))
        sys.stdout.flush()
    # Finally print the result!
    test_label_predictions, test_color_predictions, test_charSet_predictions = eval_in_batches(test_data, sess,
                               eval_label_prediction, eval_color_prediction, eval_charset_prediction, eval_data_node)
    test_label_error = error_rate(test_label_predictions, test_labels)
    test_color_error = error_rate(test_color_predictions, test_colors)
    test_charSet_error = error_rate(test_charSet_predictions, test_charSet)
    print('Test label error: %.1f%%' % test_label_error)
    print('Test color error: %.1f%%' % test_color_error)
    print('Test charset error: %.1f%%' % test_charSet_error)
    if FLAGS.self_test:
      print('test_label_error', test_label_error)
      assert test_label_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
        test_label_error,)
    train_writer.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--use_fp16',
      default=False,
      help='Use half floats instead of full floats if True.',
      action='store_true'
  )
  parser.add_argument(
      '--self_test',
      default=False,
      action='store_true',
      help='True if running a self test.'
  )
  parser.add_argument('--summaries_dir', type=str, default='log',
                      help='Summaries directory')
  FLAGS = parser.parse_args()

  tf.app.run()
