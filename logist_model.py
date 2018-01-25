from collections import namedtuple

import numpy as np
import tensorflow as tf
import six
import math

from tensorflow.python.training import moving_averages
FLAGS = tf.app.flags.FLAGS
tf.flags.DEFINE_integer("hidden_units", 100,
                     "Number of units in the hidden layer of the NN")


class LRNet(object):
  def __init__(self, images, labels, mode):
    """ResNet constructor.

    Args:
      hps: Hyperparameters.
      images: Batches of images. [batch_size, image_size, image_size, 3]
      labels: Batches of labels. [batch_size, num_classes]
      mode: One of 'train' and 'eval'.
    """
    self._images = images
    self.labels = labels
    self.mode = mode

    self._extra_train_ops = []

  def build_graph(self):
    IMAGE_PIXELS = FLAGS.image_size
    self.global_step = tf.Variable(0, name="global_step", trainable=False)

    # Variables of the hidden layer
    hid_w = tf.Variable(
        tf.truncated_normal(
            [IMAGE_PIXELS * IMAGE_PIXELS * 3, FLAGS.hidden_units],
            stddev=1.0 / IMAGE_PIXELS),
        name="hid_w")
    hid_b = tf.Variable(tf.zeros([FLAGS.hidden_units]), name="hid_b")

    # Variables of the softmax layer
    sm_w = tf.Variable(
        tf.truncated_normal(
            [FLAGS.hidden_units, 10],
            stddev=1.0 / math.sqrt(FLAGS.hidden_units)),
        name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    # Ops: located on the worker specified with FLAGS.task_index
    x = tf.reshape(self._images, [-1, IMAGE_PIXELS * IMAGE_PIXELS * 3]) #tf.placeholder(tf.float32, [None, IMAGE_PIXELS * IMAGE_PIXELS * 3])
    y_ = self.labels #tf.placeholder(tf.float32, [None, 10])

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    self.predictions = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))
    cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(self.predictions, 1e-10, 1.0)))

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    if FLAGS.sync_replicas:
        replicas_to_aggregate = FLAGS.replicas_to_aggregate

        optimizer = tf.train.SyncReplicasOptimizer( optimizer,
          replicas_to_aggregate=FLAGS.replicas_to_aggregate,
          total_num_replicas=FLAGS.replicas_to_aggregate,
          name="mnist_sync_replicas")

    train_step = optimizer.minimize(cross_entropy, global_step=self.global_step)
    self.train_op = train_step 
    self.cost = cross_entropy 

    is_chief = (FLAGS.task_index == 0)
    if FLAGS.sync_replicas:
      self.local_init_op = optimizer.local_step_init_op
      if is_chief:
        self.local_init_op = optimizer.chief_init_op

      self.ready_for_local_init_op = optimizer.ready_for_local_init_op

      # Initial token and chief queue runners required by the sync_replicas mode
      self.chief_queue_runner = optimizer.get_chief_queue_runner()
      self.sync_init_op = optimizer.get_init_tokens_op()

    self.init_op = tf.global_variables_initializer()







