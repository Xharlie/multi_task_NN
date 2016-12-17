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


WORK_DIRECTORY = 'data_2task'
IMAGE_SIZE = 56
ORI_IMAGE_SIZE = 28
NUM_CHANNELS = 3
ORI_NUM_CHANNELS = 1
PIXEL_DEPTH = 255
NUM_LABELS = 10
NUM_CHARSET = 2
SEED = 66478  # Set to None for random seed.
BATCH_SIZE = 64
NUM_EPOCHS = 100
EVAL_BATCH_SIZE = 64
EVAL_FREQUENCY = 50  # Number of steps between evaluations.
NUM_COLORS = 3
NUM_TYPE = 2
LABEL_LOSS_WEIGHT = 0.35
COLOR_LOSS_WEIGHT = 0.05
CHARSET_LOSS_WEIGHT = 0.6
REGULRATE = 5e-5
PARA_INIT_STD = 0.1
PARA_INIT_CONS = 0.1
BASE_LRN_RATE = 0.005
NUM_SAMPLES = 70000
TRAINING_SIZE = 55000
VALIDATION_SIZE = 5000
TEST_SIZE = 10000
# FLAGS = None
