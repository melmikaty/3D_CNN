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
""" This module contains functions related to the networt architecture.
    Some global variables are dataset dependent.
    Subroutines are dependent on the way the data and labels are saved.

    Update configure.py if using a new dataset.
"""

# External Packages and Modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import re
import sys

from six.moves import urllib
import tensorflow as tf

import in_data
import configure

FLAGS = tf.app.flags.FLAGS

# Globals
# Read configuration file
CFG = configure.cfg # read configuration file
batch_size = CFG['batch_size']
data_dir   = CFG['data_dir']

NUM_CLASSES                      = CFG['nClass']
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = CFG['nTrain']
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL  = CFG['nEval']

MOVING_AVERAGE_DECAY       = CFG['MOVING_AVERAGE_DECAY']
NUM_EPOCHS_PER_DECAY       = CFG['NUM_EPOCHS_PER_DECAY']
LEARNING_RATE_DECAY_FACTOR = CFG['LEARNING_RATE_DECAY_FACTOR']
INITIAL_LEARNING_RATE      = CFG['INITIAL_LEARNING_RATE']

tf.app.flags.DEFINE_integer('batch_size',batch_size,
                                    "Number of images to process in a batch.")
tf.app.flags.DEFINE_string('data_dir',data_dir,
                                    """Path to the data directory.""")
tf.app.flags.DEFINE_boolean('use_fp16',False,"""Train the model using fp16.""")

TOWER_NAME = 'tower'

""" CODE STARTS HERE """

def activation_summary(x):
    """
    Args:
        x -> tensor
    Rtns:
        nothing
    """
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def variable_on_cpu(name, shape, initializer):
    """
    Args:
        name        -> name of the variable
        shape       -> list of ints
        initializer -> initializer for variable
    Rtns:
        var         -> variable tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
        var   = tf.get_variable(name,shape,initializer=initializer,dtype=dtype)
    return var

def variable_with_weight_decay(name, shape, stddev, wd):
    """
    Note that the Variable is initialized with a truncated normal distribution.
    A weight decay is added only if one is specified.
    Args:
        name   -> name of the variable
        shape  -> list of ints
        stddev -> standard deviation of a truncated Gaussian
        wd     -> add L2Loss weight decay multiplied by this float.
                        If None, weight decay is not added for this Variable.
    Rtns:
        var    -> variable tensor
    """
    dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
    var   = variable_on_cpu(name,shape,
                    tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
    if wd is not None:
        weight_decay = tf.mul(tf.nn.l2_loss(var),wd,name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var

def inputs_train():
    """
    Args:
        nothing
    Rtns:
        img3_batch  -> 5D float32 or float16 tensor of [batch_size,h,w,d,c]
        label_batch -> 1D float32 or float16 tensor of [batch_size]
    Raises:
        ValueError -> If no data_dir
    """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir                = os.path.join(FLAGS.data_dir)
    img3_batch, label_batch = in_data.inputs_train(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        img3_batch  = tf.cast(img3_batch, tf.float16)
        label_batch = tf.cast(label_batch, tf.float16)
    return img3_batch, label_batch

def inputs_eval():
    """
    Args:
        nothing
    Rtns:
        img3_batch  -> 5D float32 or float16 tensor of [batch_size,h,w,d,c]
        label_batch -> 1D float32 or float16 tensor of [batch_size]
    Raises:
        ValueError -> If no data_dir
      """
    if not FLAGS.data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir               = os.path.join(FLAGS.data_dir)
    img3_batch, label_batch = in_data.inputs_eval(data_dir=data_dir,
                                                    batch_size=FLAGS.batch_size)

    if FLAGS.use_fp16:
        img3_batch   = tf.cast(img3_batch, tf.float16)
        label_batch  = tf.cast(label_batch, tf.float16)
    return img3_batch, label_batch


def inference(img3_batch):
    """
    Neural network model.
    keys:
        h -> height, w -> width, d -> depth, c -> channel
    Args:
        img_batch -> image batch returned from distorted_inputs() or inputs()
    Rtns:
        logits    -> output of the softmax classifier
    """
    """
    Current architecture:
    Layer 1 -> conv1 (64 filters) + pool1 + norm1
    Layer 2 -> conv2 (64 filters) + norm2 + pool2
    Layer 3 -> fcon3 with 384 neurons
    Layer 4 -> fcon4 with 192 neurons
    Softmax
    """

    ##### LAYER1 ###############################################################
    """ conv1 """
    with tf.variable_scope('conv1') as scope:
        conv_in  = img3_batch

        # Parameters
        filter_h = 3
        filter_w = 3
        filter_d = 3
        filter_c = conv_in.get_shape()[4].value
        filter_n = 64
        stride_h = 1
        stride_w = 1
        stride_d = 1

        # Computation
        kernel         = variable_with_weight_decay('weights',
                        shape=[filter_h,filter_w,filter_d,filter_c,filter_n],
                                                        stddev=5e-2,wd=0.0)
        conv           = tf.nn.conv3d(conv_in,kernel,
                                [1,stride_h,stride_w,stride_d,1],padding='SAME')
        biases         = variable_on_cpu('biases',[filter_n],
                                                tf.constant_initializer(0.0))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv1          = tf.nn.relu(pre_activation,name=scope.name)
        activation_summary(conv1)

    """ pool1 """
    # Parameters
    pool_h = 3;pool_w = 3;pool_d = 3
    stride_h = 2;stride_w = 2; stride_d = 3

    # Computation
    pool1 = tf.nn.max_pool3d(conv1,ksize=[1,pool_h,pool_w,pool_d,1],
        strides=[1,stride_h,stride_w,stride_d,1],padding='SAME',name='pool1')

    """ norm1 """
    norm1 = tf.nn.l2_normalize(pool1,dim=0,epsilon=1e-12,name='norm1')

    #### LAYER 2 ###############################################################
    """ conv2 """
    with tf.variable_scope('conv2') as scope:
        conv_in  = norm1

        # Parameters
        filter_h = 5
        filter_w = 5
        filter_d = 5
        filter_c = conv_in.get_shape()[4].value
        filter_n = 64
        stride_h = 1
        stride_w = 1
        stride_d = 1

        # Computation
        kernel         = variable_with_weight_decay('weights',
                        shape=[filter_h,filter_w,filter_d,filter_c,filter_n],
                                                            stddev=5e-2,wd=0.0)
        conv           = tf.nn.conv3d(conv_in,kernel,
                            [1,stride_h,stride_w,stride_d,1],padding='SAME')
        biases         = variable_on_cpu('biases',[filter_n],
                                                tf.constant_initializer(0.1))
        pre_activation = tf.nn.bias_add(conv,biases)
        conv2          = tf.nn.relu(pre_activation,name=scope.name)
        activation_summary(conv2)

    """ norm2 """
    norm2 = tf.nn.l2_normalize(conv2,dim=0,epsilon=1e-12,name='norm2')

    """ pool2 """
    # Parameters
    pool_h = 3;pool_w = 3; pool_d = 3
    stride_h = 2;stride_w = 2; stride_d = 2

    # Computation
    pool2 = tf.nn.max_pool3d(norm2,ksize=[1,pool_h,pool_w,pool_d,1],
        strides=[1,stride_h,stride_w,stride_d,1],padding='SAME',name='pool2')

    #### LAYER 3 ###############################################################
    """ fcon3 """
    with tf.variable_scope('fcon3') as scope:
        fcon_in = pool2

        # Parameters
        fcon_n = 384

        # ONLY FIRST FC LAYER
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(fcon_in,[FLAGS.batch_size, -1])

        # Computation
        dim     = reshape.get_shape()[1].value
        weights = variable_with_weight_decay('weights',shape=[dim,fcon_n],
                                                        stddev=0.04,wd=0.004)
        biases  = variable_on_cpu('biases',[fcon_n],
                                                tf.constant_initializer(0.1))
        fcon3   = tf.nn.relu(tf.matmul(reshape,weights)+biases,name=scope.name)
        activation_summary(fcon3)

    #### LAYER 4 ###############################################################
    """ fcon4 """
    with tf.variable_scope('fcon4') as scope:
        fcon_in = fcon3

        # Parameters
        fcon_n = 192

        # Computation
        dim     = fcon_in.get_shape()[1].value
        weights = variable_with_weight_decay('weights',shape=[dim,fcon_n],
                                                        stddev=0.04,wd=0.004)
        biases  = variable_on_cpu('biases',[fcon_n],
                                                tf.constant_initializer(0.1))
        fcon4   = tf.nn.relu(tf.matmul(fcon_in, weights)+biases,name=scope.name)
        activation_summary(fcon4)

    #### Classifier ############################################################
    """ Softmax """
    # linear layer(WX + b),
    # We don't apply softmax here because
    # tf.nn.sparse_softmax_cross_entropy_with_logits accepts the unscaled logits
    # and performs the softmax internally for efficiency.
    with tf.variable_scope('softmax_linear') as scope:
        sftmx_in = fcon4

        # Parameters
        dim    = sftmx_in.get_shape()[1].value
        # Computation
        weights        = variable_with_weight_decay('weights',
                                                [dim,NUM_CLASSES],
                                                stddev=1/float(dim),wd=0.0)
        biases         = variable_on_cpu('biases',[NUM_CLASSES],
                                                tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(sftmx_in,weights),biases,
                                                                name=scope.name)
        activation_summary(softmax_linear)

        logits         = softmax_linear

    return logits

def loss(logits, label_batch):
    """
    Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
        logits      -> logits from inference()
        label_batch -> 1D tensor of [batch_size]
    Rtns:
        total_loss  -> float tensor
    """
    # Calculate the average cross entropy loss across the batch.
    label_batch        = tf.cast(label_batch,tf.int64)
    cross_entropy      = tf.nn.sparse_softmax_cross_entropy_with_logits(logits,
                                label_batch,name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses',cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight
    # decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def add_loss_summaries(total_loss):
    """
    Generates moving average for all losses and associated summaries for
    visualizing the performance of the network.
    Args:
        total_loss       -> total loss from loss()
    Rtns:
        loss_averages_op -> op for generating moving averages of losses
    """
    # Compute the moving average of all individual losses and the total loss.
    loss_averages    = tf.train.ExponentialMovingAverage(0.9, name='avg')
    losses           = tf.get_collection('losses')
    loss_averages_op = loss_averages.apply(losses + [total_loss])

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.
    for l in losses + [total_loss]:
        # Name each loss as '(raw)' and name the moving average version of the loss
        # as the original loss name.
        tf.summary.scalar(l.op.name +' (raw)', l)
        tf.summary.scalar(l.op.name,loss_averages.average(l))
    return loss_averages_op

def train(total_loss, global_step):
    """
    Create an optimizer and apply to all trainable variables.
    Add moving average for all trainable variables.
    Args:
        total_loss  -> total loss from loss()
        global_step -> int with the number of training steps processed
    Returns:
        train_op    -> op for training
    """
    # Variables that affect learning rate.
    num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
    decay_steps           = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,global_step,
                                        decay_steps,LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
    tf.summary.scalar('learning_rate',lr)

    # Generate moving averages of all losses and associated summaries.
    loss_averages_op = add_loss_summaries(total_loss)

    # Compute gradients.
    with tf.control_dependencies([loss_averages_op]):
        opt   = tf.train.GradientDescentOptimizer(lr)
        grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    # Add histograms for gradients.
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)

    # Track the moving averages of all trainable variables.
    variable_averages     = tf.train.ExponentialMovingAverage(
                                            MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
        train_op = tf.no_op(name='train')

    return train_op
