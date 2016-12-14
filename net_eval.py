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
""" This module contains functions related to evaluating the networt.
    Some global variables are dataset dependent.
    Subroutines are dependent on the way the data and labels are saved.

    Update configure.py if using a new dataset.
"""

# External Packages and Modules
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import net_model
import configure

# Globals
# Read configuration file
CFG = configure.cfg
train_ckpt_dir = CFG['train_ckpt_dir']
eval_ckpt_dir  = CFG['eval_ckpt_dir']
eval_interval_secs = CFG['eval_interval_secs']
nEval              = CFG['nEval']

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('eval_dir',eval_ckpt_dir,
                            """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('checkpoint_dir',train_ckpt_dir,
                            """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs',eval_interval_secs,
                                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples',nEval,
                                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                                        """Whether to run eval only once.""")


""" CODE STARTS HERE """

def eval_once(saver,summary_writer,top_k_op, summary_op):
    """Run Eval once.
    Args:
        saver          -> saver
        summary_writer -> summary writer
        top_k_op       -> top K op
        summary_op     -> Summary op
    Rtns:
        nothing
    Prin:
        prints precision
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            # Restore from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            # Extract global_step
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord,
                                                    daemon=True, start=True))

            num_iter           = int(math.ceil(
                                        FLAGS.num_examples / FLAGS.batch_size))
            true_count         = 0  # Counts the number of correct predictions.
            total_sample_count = num_iter * FLAGS.batch_size
            step               = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step       += 1

            # Compute precision @ 1.
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    """
    Evaluate test data
    Args:
        nothing
    Rtns:
        nothing
    Prin:
            prints precision
    """
    with tf.Graph().as_default() as g:
        # Get img3 and labels from evaluation dataset.
        img3_batch, label_batch = net_model.inputs_eval()

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = net_model.inference(img3_batch)
        # Calculate predictions.
        top_k_op = tf.nn.in_top_k(logits,label_batch, 1)

        # Restore the moving average version of the learned variables for eval.
        variable_averages    = tf.train.ExponentialMovingAverage(
                                                net_model.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver                = tf.train.Saver(variables_to_restore)

        # Build the summary operation based on the TF collection of Summaries.
        summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):  # pylint: disable=unused-argument
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()

if __name__ == '__main__':
  tf.app.run()
