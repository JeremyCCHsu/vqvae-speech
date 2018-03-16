from os import listdir
from os.path import isdir, join, split, splitext

import tensorflow as tf
import numpy as np

from tensorflow.contrib.lookup import index_table_from_tensor

def list_dir(path):
  ''' retrieve the 'short name' of the dirs '''
  return sorted([f for f in listdir(path) if isdir(join(path, f))])


def list_full_filenames(path):
  ''' return a generator of full filenames '''
  filenames = (
      join(path, f)
      for f in listdir(path)
      if not isdir(join(path, f))
  )
  return sorted(filenames)



def make_mu_law_speaker_length(x, speaker, text, segment):
  '''
    `x`: `np.array`. 1st dim is N, 2nd (if any) is #channels
    `texts`: bytes
  '''
  assert isinstance(speaker, str)
  assert isinstance(text, str)
  assert isinstance(segment, str)
  assert x.dtype == np.uint8
  speaker = speaker.encode('utf8')
  segment = segment.encode('utf8')
  text = text.encode('utf8')
  T = x.shape[0]
  N = len(text)
  print('  text length: {:3d}, wav length: {:7d}'.format(N, T), end='')
  return tf.train.Example(
      features=tf.train.Features(
          feature={
              'wav': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[x.tostring()])),
              'speaker': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[speaker])),
              'text': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[text])),
              'segment': tf.train.Feature(
                  bytes_list=tf.train.BytesList(value=[segment])),
              'wav_len': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[T])),
              'text_len': tf.train.Feature(
                  int64_list=tf.train.Int64List(value=[N])),
          }
      )
  )


class ByteWavReader(object):
  ''' Read TF record that serialized mu-law wav (byte) '''

  def __init__(
          self, speaker_list, file_pattern,
          T=2**12, batch_size=1, num_epoch=None,
          buffer_size=4000):
    ''' `T`: sequence length '''
    with tf.device('/cpu'):
      with tf.name_scope('ByteInputPipeline'):
        self.speaker_list = tf.constant(speaker_list)
        self.table = index_table_from_tensor(mapping=self.speaker_list)
        self.T = T

        filenames = tf.gfile.Glob(file_pattern)
        if filenames:
          print('\nData Loarder: {} files found\n'.format(len(filenames)))
        else:
          raise ValueError('No files found: {}'.format(file_pattern))

        dataset = (
            tf.data.TFRecordDataset(filenames)
            .map(self._parse_function)
            .shuffle(buffer_size)
            .batch(batch_size)
            .repeat(num_epoch)
        )
        self.iterator = dataset.make_initializable_iterator()
        self.x, self.y = self.iterator.get_next()

  def _parse_function(self, serialized_example):
    ''' 
    condition: wav len < T
             | #padding    max index   
    -----------------------------------
       true  |  T - L       0
      false  |  0           L - T
    '''
    features = {
        'wav': tf.FixedLenFeature([], dtype=tf.string),
        'speaker': tf.FixedLenFeature([], dtype=tf.string),
        'segment': tf.FixedLenFeature([], dtype=tf.string),
        'wav_len': tf.FixedLenFeature([], dtype=tf.int64),
        'text_len': tf.FixedLenFeature([], dtype=tf.int64),
    }
    parsed = tf.parse_single_example(
        serialized=serialized_example,
        features=features,
    )
    x = tf.decode_raw(parsed['wav'], tf.uint8)
    L = parsed['wav_len']
    T = self.T
    P = tf.where(
        tf.less(L, T),
        tf.cast(T - L, tf.int32),  # must
        tf.constant(0, tf.int32)  # must
    )
    M = tf.where(
        tf.less(L, T),
        tf.constant(1, tf.int64),  # must
        L - T
    )
    M = tf.cast(M, tf.int64)
    o = tf.zeros([P, ], tf.uint8)
    x = tf.concat([o, x], 0)
    i = tf.random_uniform([], 0, M, tf.int64)
    xi = tf.reshape(x[i: i + T], [T])
    return tf.cast(xi, tf.int64), self.table.lookup(parsed['speaker']),


class ByteWavWholeReader(object):
  ''' Read TF record that serialized mu-law wav (byte) '''

  def __init__(self, speaker_list, filenames, num_epoch=1):
    with tf.device('/cpu'):
      with tf.name_scope('ByteInputPipeline'):
        self.speaker_list = tf.constant(speaker_list)
        self.table = index_table_from_tensor(mapping=self.speaker_list)

        print('{} files found'.format(len(filenames)))
        dataset = (
            tf.data.TFRecordDataset(filenames)
            .map(self._parse_function)
            .batch(1)
            .repeat(num_epoch)
        )

        self.iterator = dataset.make_initializable_iterator()
        self.x, self.y, self.f, self.w, self.t = self.iterator.get_next()

  def _parse_function(self, serialized_example):
    features = {
        'wav': tf.FixedLenFeature([], dtype=tf.string),
        'speaker': tf.FixedLenFeature([], dtype=tf.string),
        'segment': tf.FixedLenFeature([], dtype=tf.string),
        'wav_len': tf.FixedLenFeature([], dtype=tf.int64),
        'text_len': tf.FixedLenFeature([], dtype=tf.int64),
    }
    parsed = tf.parse_single_example(
        serialized=serialized_example,
        features=features,
    )
    x = tf.decode_raw(parsed['wav'], tf.uint8)
    return tf.cast(x, tf.int64), self.table.lookup(parsed['speaker']), parsed['segment'], parsed['wav_len'], parsed['text_len']
