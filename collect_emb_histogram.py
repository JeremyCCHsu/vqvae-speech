'''
TODO: 1. Keyboard interruption to save now
      2. Save periodically
      3. Validate periodically. (estimate the production time)
'''

from datetime import datetime
from glob import glob
from os.path import join

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models.vqvae import VQVAE
from util.wrapper import load, json2dict, txt2list
from dataloader.vctk import ByteWavWholeReader


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'file_pattern',
    'dataset/VCTK/tfr/*/*.tfr',
    'File patterns of text corpora')
tf.app.flags.DEFINE_string(
  'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
  'ckpt', None, 'model checkpoint name, e.g. model.ckpt-398897')
tf.app.flags.DEFINE_string(
  'speaker_list', './etc/speakers.tsv', 'List of global control signal')
tf.app.flags.DEFINE_string('mode', 'id', 'Mode: exemplar, encoding, id')


def main(unused_args):
  if args.logdir is None:
    raise ValueError('Please specify the dir to the checkpoint')

  arch = tf.gfile.Glob(join(args.logdir, 'arch*.json'))[0]
  arch = json2dict(arch)

  net = VQVAE(arch)

  data = ByteWavWholeReader(
    speaker_list=txt2list(args.speaker_list),
    filenames=tf.gfile.Glob(args.file_pattern))

  ZH = net.encode(data.x, args.mode)

  ema = tf.train.ExponentialMovingAverage(decay=0.995)
  trg_vars = {ema.average_name(v): v for v in tf.trainable_variables()}
  saver = tf.train.Saver(trg_vars)


  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(allow_growth=True))
  with tf.Session(config=sess_config) as sess:
    sess.run(tf.tables_initializer())
    sess.run(data.iterator.initializer)
    sess.run(tf.global_variables_initializer())
    load(saver, sess, args.logdir, ckpt=args.ckpt)

    hist = np.zeros([arch['num_exemplar'],], dtype=np.int64)
    counter = 1
    while True:
      try:
        z_ids = sess.run(ZH)
        print('\rNum of processed files: {:d}'.format(counter), end='')
        counter += 1
        for i in z_ids[0]:  # bz = 1
          hist[i] += 1
      except tf.errors.OutOfRangeError:
        print()
        break

    with open('histogram.npf', 'wb') as fp:
      hist.tofile(fp)

    plt.figure(figsize=[10, 2])
    plt.plot(np.log10(hist + 1), '.')
    plt.xlim([0, arch['num_exemplar'] - 1])
    plt.ylabel('log-frequency')
    plt.xlabel('exemplar index')
    plt.savefig('histogram.png')
    plt.close()


if __name__ == '__main__':
  tf.app.run()
