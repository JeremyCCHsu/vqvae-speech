import os

import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.training.summary_io import SummaryWriterCache


def accuracy_of_minibatch(labels, predictions, name='accuracy'):
  ''' both inputs are `tf.int64` '''
  with tf.name_scope('MinibatchAccuracy'):
    accuracy = tf.reduce_mean(
      tf.cast(
        tf.equal(labels, predictions),
        tf.float32
      )
    )
    tf.summary.scalar(name, accuracy)
    return accuracy


def visualize_embeddings(logdir, var_list, tsv_list):
  assert len(var_list) == len(tsv_list), 'Inconsistent length of lists'

  config = projector.ProjectorConfig()
  for v, f in zip(var_list, tsv_list):
    embedding = config.embeddings.add()
    embedding.tensor_name = v.name
    if f is not None:
      _, filename = os.path.split(f)
      meta_tsv = os.path.join(logdir, filename)
      tf.gfile.Copy(f, meta_tsv)  
      embedding.metadata_path = filename  # save relative path

  writer = SummaryWriterCache.get(logdir)
  projector.visualize_embeddings(writer, config)
