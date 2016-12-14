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
""" This module contains functions related to training a network.
    Some global variables are dataset dependent.
    Subroutines are dependent on the way the data and labels are saved.

    Update configure.py if using a new dataset.
"""

# External Packages and Modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import net_model
import configure

# Globals
# Read configuration file
CFG = configure.cfg
train_ckpt_dir = CFG['train_ckpt_dir']
max_steps      = CFG['max_steps']

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir',train_ckpt_dir,
                """Directory where to write event logs and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps',max_steps,
                                    """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                                        """Whether to log device placement.""")


""" CODE STARTS HERE """

def train():
    """
    Train the model for max_steps
    Args:
        nothing
    Rtns:
        nothing
    """
    with tf.Graph().as_default():
        global_step = tf.Variable(0,trainable=False)

        img3_batch, label_batch = net_model.inputs_train()
        logits                  = net_model.inference(img3_batch)
        loss                    = net_model.loss(logits, label_batch)
        train_op                = net_model.train(loss, global_step)

        # Create a saver.
        saver = tf.train.Saver(tf.global_variables())

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        # Build an initialization operation to run below.
        init =  tf.global_variables_initializer()

        # Start running operations on the Graph.
        sess = tf.Session(config=tf.ConfigProto(
                            log_device_placement=FLAGS.log_device_placement))
        sess.run(init)

        # Start the queue runners.
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

        for step in xrange(FLAGS.max_steps):
            start_time    = time.time()
            _, loss_value = sess.run([train_op,loss])
            duration      = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size
                examples_per_sec      = num_examples_per_step / duration
                sec_per_batch         = float(duration)

                format_str = (
                '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)')
                print (format_str % (datetime.now(),step,
                                    loss_value,examples_per_sec,sec_per_batch))

            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            if step % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess,checkpoint_path,global_step=step)


def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()
