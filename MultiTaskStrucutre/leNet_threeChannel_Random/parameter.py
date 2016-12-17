from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import gzip
import os
import sys
import time

import random
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets import base


SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
WORK_DIRECTORY = 'data_2task'
IMAGE_SIZE = 28
NUM_CHANNELS = 3
ORI_NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
VALIDATION_SIZE = 5000  # Size of the validation set.
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 3
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 20  # Number of steps between evaluations.
NUM_COLORS = 3
LABEL_LOSS_WEIGHT = 1
COLOR_LOSS_WEIGHT = 0
REGULRATE = 5e-4

# FLAGS = None