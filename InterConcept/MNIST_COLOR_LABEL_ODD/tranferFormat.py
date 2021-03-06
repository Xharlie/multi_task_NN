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


def extract_data(filename, num_images):
  """Extract the images into a 4D tensor [image index, y, x, channels].

  Values are rescaled from [0, 255] down to [-0.5, 0.5].
  """
  print('Extracting', filename)
  reformData = numpy.full((num_images, IMAGE_SIZE, IMAGE_SIZE, ORI_NUM_CHANNELS), 0, dtype=numpy.float32)
  with gzip.open(filename) as bytestream:
    bytestream.read(16)
    buf = bytestream.read(ORI_IMAGE_SIZE * ORI_IMAGE_SIZE * num_images * ORI_NUM_CHANNELS)
    data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
    data = (data - (PIXEL_DEPTH / 2.0)) / PIXEL_DEPTH
    print(data.shape)
    data = data.reshape(num_images, ORI_IMAGE_SIZE, ORI_IMAGE_SIZE, ORI_NUM_CHANNELS)
    for i in range(0, num_images):
      for j in range(0, IMAGE_SIZE):
        for k in range(0, IMAGE_SIZE):
          reformData[i][j][k] = data[i][j//2][k//2]
    return reformData

def extract_labels(filename, num_images):
  """Extract the labels into a vector of int64 label IDs."""
  print('Extracting', filename)
  with gzip.open(filename) as bytestream:
    bytestream.read(8)
    buf = bytestream.read(1 * num_images)
    labels = numpy.frombuffer(buf, dtype=numpy.uint8)
  return labels

def write_back(file, data):
  print('Writing', file)
  data = data.ravel()
  f = open(file, "wb")
  f.write(data.tobytes())
  f.close()

def transformImage_irrelevent(images, labels):
  colored_images = numpy.zeros((images.shape[0], images.shape[1], images.shape[2], 3), dtype=numpy.float32)
  color = numpy.zeros(shape=labels.size, dtype=numpy.uint8)
  charSet = numpy.zeros(shape=labels.size, dtype=numpy.uint8)
  for i in range(images.shape[0]):
    index = random.randint(0,2)
    color[i] = index
    charSet[i] = 0 if (labels[i] % 2 == 0) else 1
    for j in range(images.shape[1]):
      for k in range(images.shape[2]):
        redBase = random.uniform(0, 0.5)
        greenBase = random.uniform(0, 0.5)
        blueBase = random.uniform(0, 0.5)
        colored_images[i][j][k][0] = redBase
        colored_images[i][j][k][1] = greenBase
        colored_images[i][j][k][2] = blueBase
        colored_images[i][j][k][index] += images[i][j][k] * 0.5
  return colored_images, color, charSet

def uniqueCharacter():
  s = open("plain-characters.dat", "r").read()
  print("Unique Characters: {%s}" % ''.join(set(s)))

if __name__ == '__main__':
  data = numpy.concatenate((extract_data(WORK_DIRECTORY+"/train-images-idx3-ubyte.gz", 60000),
                            extract_data(WORK_DIRECTORY+"/t10k-images-idx3-ubyte.gz", 10000)), axis=0)
  label = numpy.concatenate((extract_labels(WORK_DIRECTORY+"/train-labels-idx1-ubyte.gz", 60000),
                             extract_labels(WORK_DIRECTORY+"/t10k-labels-idx1-ubyte.gz", 10000)), axis=0)
  colored_images, color, charSet = transformImage_irrelevent(data, label)
  write_back(WORK_DIRECTORY+"/image_source.dat", colored_images)
  write_back(WORK_DIRECTORY+"/label_source.dat", label)
  write_back(WORK_DIRECTORY+"/charSet_source.dat", charSet)
  write_back(WORK_DIRECTORY+"/color_source.dat", color)

