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
# #############################################################################
# This is an implementaion of Convolutional Neural Netwroks for 3D volumes.
# The code is a modification of the code provided in TensorFlow tutorials for
# CIFAR-10 dataset.
# For help, please refer to the 'readme' file
# The code is modifed by: Mohamed ElMikaty
# Last update: 14 Dec 16
################################################################################
""" This module contains functions related to reading a given dataset.
    Some global variables are dataset dependent.
    Subroutines are dependent on the way the data and labels are saved.

    Update configure.py if using a new dataset.
"""

# External Packages and Modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from six.moves import xrange
import tensorflow as tf

import configure

# Globals
# Read configuration file
CFG = configure.cfg
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = CFG['nTrain']
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL  = CFG['nEval']

""" CODE STARTS HERE """

def read_data(file_queue):
    """
    Data is saved in binary files.
    Each row has:
        1st byte      -> label
        2nd-last byte -> 3D Volume [height, width, depth, channels]
    Args:
        file_queue      -> a queue of file names saved as strings
    Rtns:
        An object with:
            height      -> volume height
            width       -> volume width
            depth       -> volume depth
            nChan       -> number of channels
            key         -> scalar tensor with file name and record number
            label       -> 1D int32 tensor with the associated label
            img3_uint8  -> 4D uint8 tensor with image data
    """

    class record_data(object):
        pass
    img3_obj = record_data()

    # Dimensions of data
    label_bytes     = 1
    img3_obj.height = CFG['height']
    img3_obj.width  = CFG['width']
    img3_obj.depth  = CFG['depth']
    img3_obj.nChan  = CFG['nChan']

    # Size in memory
    img3_bytes   = img3_obj.height*img3_obj.width*img3_obj.depth*img3_obj.nChan
    record_bytes = label_bytes + img3_bytes

    # Read a record
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    img3_obj.key,value = reader.read(file_queue)

    # Convert from a string to a vector of uint8 that is record_bytes long
    record_bytes = tf.decode_raw(value, tf.uint8)

    # First byte represent the label, which we convert from uint8 -> int32
    img3_obj.label = tf.cast(tf.slice(record_bytes,[0],[label_bytes]),tf.int32)

    # Remaining bytes after the label represent the image, which we reshape from
    # [depth * height * width] to [depth,height, width]
    img3_obj.img3_uint8 = tf.reshape(tf.slice(record_bytes, [label_bytes],
                [img3_bytes]),[img3_obj.height,img3_obj.width,img3_obj.depth,
                                                                img3_obj.nChan])

    return img3_obj

def generate_image_and_label_batch(img3,label,min_queue_examples,
                                    batch_size,shuffle):
    """
    Args:
        img                -> 4D float32 tensor with image data
        label              -> 1D int32 tensor with the associated label
        min_queue_examples -> int32 with minimum number of image samples
        batch_size         -> int32 with number of images per batch
        shuffle            -> boolean to use shuffling
    Rtns:
        img_batch          -> 5D float32 tensor of [batch_size,h,w,d,c]
        label_batch        -> 1D float32 tensor of [batch_size]
    """

    num_preprocess_threads = 16
    if shuffle:
        img3_batch, label_batch = tf.train.shuffle_batch(
                                [img3,label],
                                batch_size=batch_size,
                                num_threads=num_preprocess_threads,
                                capacity=min_queue_examples + 3 * batch_size,
                                min_after_dequeue=min_queue_examples
                                )
    else:
        img3_batch, label_batch = tf.train.batch(
                                [img3, label],
                                batch_size=batch_size,
                                num_threads=num_preprocess_threads,
                                capacity=min_queue_examples + 3 * batch_size
                                )

    return img3_batch, tf.reshape(label_batch,[batch_size])

def inputs_train(data_dir,batch_size):
    """
    Args:
        data_dir    -> full path to training data directory
        batch_size  -> int32 with number of images per batch
    Rtns:
        img3_batch   -> 5D float32 tensor of [batch_size,h,w,d,c]
        label_batch  -> 1D float32 tensor of [batch_size]
    """
    file_names = [os.path.join(data_dir,'data_batch_%d.bin' % i)
                    for i in xrange(1,CFG['nBatchBin']+1)] # data stored in six batches
    for f in file_names:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    # Create a queue that produces the filenames to read
    file_queue = tf.train.string_input_producer(file_names)

    # Read examples from files in the filename queue
    img3_obj     = read_data(file_queue)
    img3_reshape = tf.cast(img3_obj.img3_uint8, tf.float32)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples                = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN*
                                            min_fraction_of_examples_in_queue)
    print ('Filling queue with %d images.'
                        'This will take a few minutes.' % min_queue_examples)

    # Generate a batch of images and labels by building up a queue of examples.
    return generate_image_and_label_batch(img3_reshape, img3_obj.label,
                                    min_queue_examples,batch_size,shuffle=True)



def inputs_eval(data_dir,batch_size):
    """
    Args:
        data_dir    -> full path to evaluation data directory
        batch_size  -> int32 with number of images per batch
    Rtns:
        img3_batch  -> 5D float32 tensor of [batch_size,h,w,d,c]
        label_batch -> 1D float32 tensor of [batch_size]
    """

    file_names = [os.path.join(data_dir,'data_batch_eval.bin')]
    for f in file_names:
        if not tf.gfile.Exists(f):
            raise ValueError('Failed to find file:' + f)

    # Create a queue that produces the filenames to read.
    file_queue = tf.train.string_input_producer(file_names)

    # Read examples from files in the filename queue.
    img3_obj     = read_data(file_queue)
    img3_reshape = tf.cast(img3_obj.img3_uint8, tf.float32)

    # Ensure that the random shuffling has good mixing properties.
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples                = int(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL *
                                            min_fraction_of_examples_in_queue)

    print ('Filling queue with %d images.'
                        'This will take a few minutes.' % min_queue_examples)
    # Generate a batch of images and labels by building up a queue of examples.
    return generate_image_and_label_batch(img3_reshape,img3_obj.label,
                                min_queue_examples,batch_size,shuffle=False)
