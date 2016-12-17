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
# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from parameter import *
import argparse
import gzip
import os
import sys
import time

import numpy
import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base

def data_type(FLAGS):
  """Return the type of the activations, weights, and placeholder variables."""
  if FLAGS.use_fp16:
    return tf.float16
  else:
    return tf.float32
#
#
# def maybe_download(filename):
#   """Download the data from Yann's website, unless it's already here."""
#   if not tf.gfile.Exists(WORK_DIRECTORY):
#     tf.gfile.MakeDirs(WORK_DIRECTORY)
#   filepath = os.path.join(WORK_DIRECTORY, filename)
#   if not tf.gfile.Exists(filepath):
#     filepath, _ = urllib.request.urlretrieve(SOURCE_URL + filename, filepath)
#     with tf.gfile.GFile(filepath) as f:
#       size = f.size()
#     print('Successfully downloaded', filename, size, 'bytes.')
#   return filepath

def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  with open(filename) as bytestream:
    buf = bytestream.read(IMAGE_SIZE * IMAGE_SIZE * num_images * NUM_CHANNELS * 2)
    data = numpy.frombuffer(buf, dtype=numpy.float16).astype(numpy.float32)
    data = data.reshape(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
    return data


def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with open(filename) as bytestream:
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.int64)
  return labels


def fake_data(num_images):
  """Generate a fake dataset that matches the dimensions of MNIST."""
  data = numpy.ndarray(
      shape=(num_images, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS),
      dtype=numpy.float32)
  labels = numpy.zeros(shape=(num_images,), dtype=numpy.int64)
  for image in xrange(num_images):
    label = image % 2
    data[image, :, :, 0] = label - 0.5
    labels[image] = label
  return data, labels


def error_rate(predictions, labels):
  """Return the error rate based on dense predictions and sparse labels."""
  return 100.0 - (
      100.0 *
      numpy.sum(numpy.argmax(predictions, 1) == labels) /
      predictions.shape[0])

def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def transformImage(images, labels):
  colored_images = numpy.zeros((images.shape[0], images.shape[1], images.shape[2], 3), dtype=numpy.float32)
  color = numpy.zeros((labels.size,), dtype=numpy.int32)
  for i in range(images.shape[0]):
    index = labels[i] // 3
    if index == 3: index = 2
    color[i] = index
    for j in range(images.shape[1]):
      for k in range(images.shape[2]):
        redBase = random.uniform(0, 0.7)
        greenBase = random.uniform(0, 0.7)
        blueBase = random.uniform(0, 0.7)
        colored_images[i][j][k][0] = redBase
        colored_images[i][j][k][1] = greenBase
        colored_images[i][j][k][2] = blueBase
        colored_images[i][j][k][index] += images[i][j][k] * 0.3
  return colored_images, color
