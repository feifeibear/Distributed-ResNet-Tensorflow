# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""ResNet Train/Eval module.
"""
import time
import six
import tempfile
import sys
import os

import cifar_input
import numpy as np
import resnet_model
import logist_model
import vgg_preprocessing
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS
flags = tf.app.flags
flags.DEFINE_string('dataset', 'imagenet', 'cifar10 or cifar100 or imagenet.')
flags.DEFINE_string('mode', 'train', 'train or eval.')
flags.DEFINE_string('train_data_path', '',
                           'Filepattern for training data.')
flags.DEFINE_string('eval_data_path', '',
                           'Filepattern for eval data')
flags.DEFINE_integer('image_size', 32, 'Image side length.')
tf.app.flags.DEFINE_string('train_dir', '',
                           'Directory to keep training outputs.')
flags.DEFINE_string('eval_dir', '',
                           'Directory to keep eval outputs.')
flags.DEFINE_integer('eval_batch_count', 50,
                            'Number of batches to eval.')
flags.DEFINE_bool('eval_once', False,
                         'Whether evaluate the model only once.')
flags.DEFINE_string('log_root', '',
                           'Directory to keep the checkpoints. Should be a '
                           'parent directory of FLAGS.train_dir/eval_dir.')
flags.DEFINE_integer('num_gpus', 0,
                            'Number of gpus used for training. (0 or 1)')

flags.DEFINE_integer("task_index", None,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the master worker task the performs the variable "
                     "initialization ")
flags.DEFINE_integer("replicas_to_aggregate", None,
                     "Number of replicas to aggregate before parameter update"
                     "is applied (For sync_replicas mode only; default: "
                     "num_workers)")
flags.DEFINE_integer("train_steps", 200,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("num_epochs", 90,
                     "Number of (global) training steps to perform")
flags.DEFINE_integer("batch_size", 128, "Training batch size")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_boolean("sync_replicas", False,
                     "Use the sync_replicas (synchronized replicas) mode, "
                     "wherein the parameter updates from workers are aggregated "
                     "before applied to avoid stale gradients")
flags.DEFINE_boolean(
    "existing_servers", False, "Whether servers already exists. If True, "
    "will use the worker hosts via their GRPC URLs (one client process "
    "per worker host). Otherwise, will create an in-process TensorFlow "
    "server.")
flags.DEFINE_string("ps_hosts","localhost:2222",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", "localhost:2223,localhost:2224",
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("job_name", None,"job name: worker or ps")
flags.DEFINE_string('data_format', 'channels_first',
                           'channels_first for cuDNN, channels_last for MKL')
flags.DEFINE_integer("num_intra_threads", 0,
                     "Number of threads to use for intra-op parallelism. If set" 
                     "to 0, the system will pick an appropriate number.")
flags.DEFINE_integer("num_inter_threads", 0,
                     "Number of threads to use for inter-op parallelism. If set" 
                     "to 0, the system will pick an appropriate number.")



FLAGS = flags.FLAGS


_DEFAULT_IMAGE_SIZE = 224
_NUM_CHANNELS = 3
_LABEL_CLASSES = 1001

_MOMENTUM = 0.9
_WEIGHT_DECAY = 1e-4

_NUM_IMAGES = {
    'train': 1281167,
    'validation': 50000,
}

_FILE_SHUFFLE_BUFFER = 1024
_SHUFFLE_BUFFER = 1500

def create_config_proto():
  """Returns session config proto.
  Args:
    params: Params tuple, typically created by make_params or
            make_params_from_flags.
  """
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  if(FLAGS.num_intra_threads != 0):
      config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  if(FLAGS.num_inter_threads != 0):
      config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  # config.gpu_options.force_gpu_compatible = params.force_gpu_compatible
  # if params.gpu_memory_frac_for_testing > 0:
  #   config.gpu_options.per_process_gpu_memory_fraction = (
  #       params.gpu_memory_frac_for_testing)
  # if params.xla:
  #   config.graph_options.optimizer_options.global_jit_level = (
  #       tf.OptimizerOptions.ON_1)
  # if params.enable_layout_optimizer:
  #   config.graph_options.rewrite_options.layout_optimizer = (
  #       rewriter_config_pb2.RewriterConfig.ON)

  return config




def filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'train-%05d-of-01024' % i)
        for i in range(1024)]
  else:
    return [
        os.path.join(data_dir, 'validation-%05d-of-00128' % i)
        for i in range(128)]



def record_parser(value, is_training):
  """Parse an ImageNet record from `value`."""
  keys_to_features = {
      'image/encoded':
          tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format':
          tf.FixedLenFeature((), tf.string, default_value='jpeg'),
      'image/class/label':
          tf.FixedLenFeature([], dtype=tf.int64, default_value=-1),
      'image/class/text':
          tf.FixedLenFeature([], dtype=tf.string, default_value=''),
      'image/object/bbox/xmin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax':
          tf.VarLenFeature(dtype=tf.float32),
      'image/object/class/label':
          tf.VarLenFeature(dtype=tf.int64),
  }

  parsed = tf.parse_single_example(value, keys_to_features)

  image = tf.image.decode_image(
      tf.reshape(parsed['image/encoded'], shape=[]),
      _NUM_CHANNELS)
  image = tf.image.convert_image_dtype(image, dtype=tf.float32)

  image = vgg_preprocessing.preprocess_image(
      image=image,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      is_training=is_training)

  label = tf.cast(
      tf.reshape(parsed['image/class/label'], shape=[]),
      dtype=tf.int32)

  return image, tf.one_hot(label, _LABEL_CLASSES)


def input_fn(is_training, data_dir, batch_size, num_epochs=1):
  """Input function which provides batches for train or eval."""
  dataset = tf.contrib.data.Dataset.from_tensor_slices(filenames(is_training, data_dir))

  if is_training:
    dataset = dataset.shuffle(buffer_size=_FILE_SHUFFLE_BUFFER)

  dataset = dataset.flat_map(tf.contrib.data.TFRecordDataset)
  dataset = dataset.map(lambda value: record_parser(value, is_training),
                       num_threads=5,
                       output_buffer_size=batch_size)
  # dataset = dataset.prefetch(batch_size)

  if is_training:
    # When choosing shuffle buffer sizes, larger sizes result in better
    # randomness, while smaller sizes have better performance.
    dataset = dataset.shuffle(buffer_size=_SHUFFLE_BUFFER)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)

  iterator = dataset.make_one_shot_iterator()
  images, labels = iterator.get_next()
  return images, labels

def train(hps, server):
  """Training loop."""
  # Ops : on every worker   
  # a imagent reader get images and labels
  # images, labels = cifar_input.build_input(
  #     FLAGS.dataset, FLAGS.train_data_path, hps.batch_size, FLAGS.mode)
  images, labels = input_fn(True, FLAGS.train_data_path, FLAGS.batch_size, FLAGS.num_epochs)

  model = resnet_model.ResNet(hps, images, labels, FLAGS.mode)
  # model = logist_model.LRNet(images, labels, FLAGS.mode)
  model.build_graph()

  #param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
  #    tf.get_default_graph(),
  #    tfprof_options=tf.contrib.tfprof.model_analyzer.
  #        TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
  #sys.stdout.write('total_params: %d\n' % param_stats.total_parameters)

  #tf.contrib.tfprof.model_analyzer.print_model_analysis(
  #    tf.get_default_graph(),
  #    tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)

  truth = tf.argmax(model.labels, axis=1)
  predictions = tf.argmax(model.predictions, axis=1)
  precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

  summary_hook = tf.train.SummarySaverHook(
      save_steps=100,
      output_dir=FLAGS.train_dir,
      summary_op=tf.summary.merge([model.summaries,
                                   tf.summary.scalar('Precision', precision)]))

  logging_hook = tf.train.LoggingTensorHook(
      tensors={'step': model.global_step,
               'loss': model.cost,
               'precision': precision},
      every_n_iter=40)

  class _LearningRateSetterHook(tf.train.SessionRunHook):
    """Sets learning_rate based on global step."""

    def begin(self):
      self._lrn_rate = 0.4

    def before_run(self, run_context):
      return tf.train.SessionRunArgs(
          model.global_step,  # Asks for global step value.
          feed_dict={model.lrn_rate: self._lrn_rate})  # Sets learning rate

    def after_run(self, run_context, run_values):
      #intel resnet_50_8_nodes version
      train_step = run_values.results

      if train_step < 6240:
        self._lrn_rate = 0.1 + 0.3*train_step/6240.0
      elif train_step < 37440:
        self._lrn_rate = 0.4
      elif train_step < 74880:
        self._lrn_rate = 0.1 * 0.4
      elif train_step < 99840:
        self._lrn_rate = 0.01 * 0.4
      else:
        self._lrn_rate = 0.001 * 0.4

  is_chief = (FLAGS.task_index == 0)
  with tf.train.MonitoredTrainingSession(
      master=server.target,
      is_chief=is_chief,
      checkpoint_dir=FLAGS.log_root,
      hooks=[tf.train.StopAtStepHook(last_step=FLAGS.train_steps),
          logging_hook, _LearningRateSetterHook()],
      chief_only_hooks=[model.replicas_hook, summary_hook],
      # Since we provide a SummarySaverHook, we need to disable default
      # SummarySaverHook. To do that we set save_summaries_steps to 0.
      save_summaries_steps=0,
      stop_grace_period_secs=120,
      config=create_config_proto()) as mon_sess:
    while not mon_sess.should_stop():
      mon_sess.run(model.train_op)

def main(_):
  if FLAGS.dataset == 'imagenet':
    num_classes = 1001

  hps = resnet_model.HParams(num_classes=num_classes,
                             lrn_rate=0.1,
                             weight_decay_rate=0.0001,
                             optimizer='mom')

  # add cluster information
  if FLAGS.job_name is None or FLAGS.job_name == "":
    raise ValueError("Must specify an explicit `job_name`")
  if FLAGS.task_index is None or FLAGS.task_index =="":
    raise ValueError("Must specify an explicit `task_index`")

  print("job name = %s" % FLAGS.job_name)
  print("task index = %d" % FLAGS.task_index)

  #Construct the cluster and start the server
  ps_spec = FLAGS.ps_hosts.split(",")
  worker_spec = FLAGS.worker_hosts.split(",")

  # Get the number of workers.
  num_workers = len(worker_spec)
  FLAGS.replicas_to_aggregate = num_workers

  cluster = tf.train.ClusterSpec({
      "ps": ps_spec,
      "worker": worker_spec})

  if not FLAGS.existing_servers:
    # Not using existing servers. Create an in-process server.
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
      server.join()

  if FLAGS.num_gpus > 0:
    # Avoid gpu allocation conflict: now allocate task_num -> #gpu
    # for each worker in the corresponding machine
    gpu = (FLAGS.task_index % FLAGS.num_gpus)
    worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
  elif FLAGS.num_gpus == 0:
    # Just allocate the CPU to worker server
    cpu = 0
    worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

  with tf.device(
      tf.train.replica_device_setter(
          worker_device=worker_device,
          # ps_device="/job:ps/cpu:0",
          cluster=cluster)):

    if FLAGS.mode == 'train':
      train(hps, server)

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
