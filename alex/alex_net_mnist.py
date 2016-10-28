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

"""Timing benchmark for AlexNet inference.
To run, use:
  bazel run -c opt --config=cuda \
      third_party/tensorflow/models/image/alexnet:alexnet_benchmark
Across 100 steps on batch size = 128.
Forward pass:
Run on Tesla K40c: 145 +/- 1.5 ms / batch
Run on Titan X:     70 +/- 0.1 ms / batch
Forward-backward pass:
Run on Tesla K40c: 480 +/- 48 ms / batch
Run on Titan X:    244 +/- 30 ms / batch
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from datetime import datetime
import math
import time
import sys
import numpy
import gzip
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

FLAGS = None
SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data'
IMAGE_SIZE = 28
NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 10
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 100  # Number of steps between evaluations.
DATA_TYPE = tf.float32

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images):
  """Build the AlexNet model.
  Args:
    images: Images Tensor
  Returns:
    pool5: the last Tensor in the convolutional component of AlexNet.
    parameters: a list of Tensors corresponding to the weights and biases of the
        AlexNet model.
  """
  parameters = []
  # conv1
  with tf.name_scope('conv1') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS, 64], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv1 = tf.nn.relu(bias, name=scope)
    print_activations(conv1)
    parameters += [kernel, biases]

  # lrn1
  # TODO(shlens, jiayq): Add a GPU version of local response normalization.

  # pool1
  pool1 = tf.nn.max_pool(conv1,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 1, 1, 1],
                         padding='VALID',
                         name='pool1')
  print_activations(pool1)

  # conv2
  with tf.name_scope('conv2') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 192], dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[192], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv2 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
  print_activations(conv2)

  # pool2
  pool2 = tf.nn.max_pool(conv2,
                         ksize=[1, 2, 2, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool2')
  print_activations(pool2)

  # conv3
  with tf.name_scope('conv3') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 192, 384],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv3 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv3)

  # conv4
  with tf.name_scope('conv4') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv4 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv4)

  # conv5
  with tf.name_scope('conv5') as scope:
    kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256],
                                             dtype=tf.float32,
                                             stddev=1e-1), name='weights')
    conv = tf.nn.conv2d(conv4, kernel, [1, 1, 1, 1], padding='SAME')
    biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                         trainable=True, name='biases')
    bias = tf.nn.bias_add(conv, biases)
    conv5 = tf.nn.relu(bias, name=scope)
    parameters += [kernel, biases]
    print_activations(conv5)

  # pool5
  pool5 = tf.nn.max_pool(conv5,
                         ksize=[1, 3, 3, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID',
                         name='pool5')
  print_activations(pool5)
  # fc1
  pool_shape = pool5.get_shape().as_list()
  reshape = tf.reshape(
    pool5,
    [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  fc1_weights = tf.Variable(  # fully connected, depth 512.
    # IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64
    tf.truncated_normal([pool_shape[1] * pool_shape[2] * pool_shape[3], 512],
                        stddev=0.1,
                        seed=SEED,
                        dtype=DATA_TYPE))
  fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=DATA_TYPE))
  fc1 = tf.nn.relu(tf.matmul(reshape, fc1_weights) + fc1_biases)
  parameters += [fc1_weights, fc1_biases]

  # fc2
  fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS],
                                                stddev=0.1,
                                                seed=SEED,
                                                dtype=DATA_TYPE))
  fc2_biases = tf.Variable(tf.constant(
    0.1, shape=[NUM_LABELS], dtype=DATA_TYPE))
  parameters += [fc2_weights, fc2_biases]
  fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases)

  return fc2, parameters



def run_training():
  # Get the sets of images and labels for training, validation, and test on MNIST.
  train_data, train_labels, validation_data, validation_labels, test_data, test_labels    \
    = load_data()
  train_size = train_data.shape[0]
  # Generate placeholders for the images and labels.
  train_images_placeholder, train_labels_placeholder = placeholder_inputs(
    BATCH_SIZE)

  # Generate placeholders for the images and labels.
  eval_images_placeholder, _ = placeholder_inputs(
    EVAL_BATCH_SIZE)

  # Build a Graph that computes predictions from the inference model.
  logits, parameters = inference(train_images_placeholder)

  # Add to the Graph the Ops for loss calculation.
  loss = getloss(logits, train_labels_placeholder)

  # L2 regularization for the fully connected parameters.
  regularizers = tf.nn.l2_loss(parameters[5][0]) + tf.nn.l2_loss(parameters[5][1])    \
      + tf.nn.l2_loss(parameters[6][0]) + tf.nn.l2_loss(parameters[6][1])
  loss += 5e-4 * regularizers

  # Add to the Graph the Ops that calculate and apply gradients.
  optimizer,learning_rate = training(loss, train_size)


  # Predictions for the current training minibatch.
  train_prediction = tf.nn.softmax(logits)

  # Predictions for the test and validation, which we'll compute less often.
  eval_prediction = tf.nn.softmax(inference(eval_images_placeholder)[0])

  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction,
            feed_dict={eval_images_placeholder: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,
            feed_dict={eval_images_placeholder: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

  # Create a local session to run the training.
  start_time = time.time()
  with tf.Session() as sess:
    # Run all the initializers to prepare the trainable parameters.
    tf.initialize_all_variables().run()
    print('Initialized!')
    # Loop through training steps.
    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      print(',Step' + str(step)),
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_images_placeholder: batch_data,
                   train_labels_placeholder: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, l, lr, predictions = sess.run(
        [optimizer, loss, learning_rate, train_prediction],
        feed_dict=feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        print('Step %d (epoch %.2f), %.1f ms' %
              (step, float(step) * BATCH_SIZE / train_size,
               1000 * elapsed_time / EVAL_FREQUENCY))
        print('Minibatch loss: %.3f, learning rate: %.6f' % (l, lr))
        print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        print('Validation error: %.1f%%' % error_rate(
          eval_in_batches(validation_data, sess), validation_labels))
        sys.stdout.flush()
    # Finally print the result!
    test_error = error_rate(eval_in_batches(test_data, sess), test_labels)
    print('Test error: %.1f%%' % test_error)
    # if FLAGS.self_test:
    #   print('test_error', test_error)
    #   assert test_error == 0.0, 'expected 0.0 test_error, got %.2f' % (
    #     test_error,)




def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def training(loss, train_size):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0, dtype=DATA_TYPE)
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
    0.01,  # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    train_size,  # Decay step.
    0.95,  # Decay rate.
    staircase=True)
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate,
                                         0.9).minimize(loss,
                                                       global_step=batch)
  return optimizer,learning_rate

def getloss(logits, labels):
  """Calculates the loss from the logits and the labels.

  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].

  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int32(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss

def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.

  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  """
  images_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return images_placeholder, labels_placeholder

def load_data():
  train_data_filename = base.maybe_download('train-images-idx3-ubyte.gz', WORK_DIRECTORY,
                                            SOURCE_URL + 'train-images-idx3-ubyte.gz')
  train_labels_filename = base.maybe_download('train-labels-idx1-ubyte.gz', WORK_DIRECTORY,
                                              SOURCE_URL + 'train-labels-idx1-ubyte.gz')
  test_data_filename = base.maybe_download('t10k-images-idx3-ubyte.gz', WORK_DIRECTORY,
                                           SOURCE_URL + 't10k-images-idx3-ubyte.gz')
  test_labels_filename = base.maybe_download('t10k-labels-idx1-ubyte.gz', WORK_DIRECTORY,
                                             SOURCE_URL + 't10k-labels-idx1-ubyte.gz')

  # Extract it into numpy arrays.
  train_data = extract_data(train_data_filename, 60000)
  train_labels = extract_labels(train_labels_filename, 60000)
  test_data = extract_data(test_data_filename, 10000)
  test_labels = extract_labels(test_labels_filename, 10000)

  # Generate a validation set.
  validation_data = train_data[:VALIDATION_SIZE, ...]
  validation_labels = train_labels[:VALIDATION_SIZE]
  train_data = train_data[VALIDATION_SIZE:, ...]
  train_labels = train_labels[VALIDATION_SIZE:]
  num_epochs = NUM_EPOCHS
  return train_data, train_labels, validation_data, validation_labels, test_data, test_labels

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int32)
  return labels

def main(_):
  run_training()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=128,
      help='Batch size.'
  )
  parser.add_argument(
      '--num_batches',
      type=int,
      default=100,
      help='Number of batches to run.'
  )
  FLAGS = parser.parse_args()

  tf.app.run()


def time_tensorflow_run(session, target, info_string):
  """Run the computation to obtain the target tensor and print timing stats.
  Args:
    session: the TensorFlow session to run the computation under.
    target: the target Tensor that is passed to the session's run() function.
    info_string: a string summarizing this run, to be printed with the stats.
  Returns:
    None
  """
  num_steps_burn_in = 10
  total_duration = 0.0
  total_duration_squared = 0.0
  for i in xrange(FLAGS.num_batches + num_steps_burn_in):
    start_time = time.time()
    _ = session.run(target)
    duration = time.time() - start_time
    if i >= num_steps_burn_in:
      if not i % 10:
        print('%s: step %d, duration = %.3f' %
              (datetime.now(), i - num_steps_burn_in, duration))
      total_duration += duration
      total_duration_squared += duration * duration
  mn = total_duration / FLAGS.num_batches
  vr = total_duration_squared / FLAGS.num_batches - mn * mn
  sd = math.sqrt(vr)
  print('%s: %s across %d steps, %.3f +/- %.3f sec / batch' %
        (datetime.now(), info_string, FLAGS.num_batches, mn, sd))


def run_benchmark():
  """Run the benchmark on AlexNet."""
  with tf.Graph().as_default():
    # Generate some dummy images.
    image_size = 224
    # Note that our padding definition is slightly different the cuda-convnet.
    # In order to force the model to start with the same activations sizes,
    # we add 3 to the image_size and employ VALID padding above.
    images = tf.Variable(tf.random_normal([FLAGS.batch_size,
                                           image_size,
                                           image_size, 3],
                                          dtype=tf.float32,
                                          stddev=1e-1))

    # Build a Graph that computes the logits predictions from the
    # inference model.
    pool5, parameters = inference(images)

    # Build an initialization operation.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'
    sess = tf.Session(config=config)
    sess.run(init)

    # Run the forward benchmark.
    time_tensorflow_run(sess, pool5, "Forward")

    # Add a simple objective so we can calculate the backward pass.
    objective = tf.nn.l2_loss(pool5)
    # Compute the gradient with respect to all the parameters.
    grad = tf.gradients(objective, parameters)
    # Run the backward benchmark.
    time_tensorflow_run(sess, grad, "Forward-backward")
