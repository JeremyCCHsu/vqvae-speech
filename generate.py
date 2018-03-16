'''
TODO: 1. Keyboard interruption to save now
      2. Save periodically
      3. Validate periodically. (estimate the production time)
'''

import os

from time import sleep
from datetime import datetime

import tensorflow as tf
import numpy as np

from models.vqvae import VQVAE

from dataloader.vctk import ByteWavWholeReader

from util.audio import mu_law_decode
from util.wrapper import load, json2dict, txt2list


args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
  'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
  'file_pattern', None, 'File patterns of text corpora'
)

tf.app.flags.DEFINE_string(
  'speaker_list', './etc/speakers.tsv', 'List of global control signal'
)
tf.app.flags.DEFINE_integer('period', 0, 'Periodically generate')
tf.app.flags.DEFINE_string('mode', 'exemplar', 'Mode: exemplar, encoding, id')
tf.app.flags.DEFINE_string(
  'ckpt', None, 'model checkpoint name, e.g. model.ckpt-398897')


def get_default_logdir(logdir_root):
  STARTED_DATESTRING = datetime.now().strftime('%Y-%0m%0d-%0H%0M')
  logdir = os.path.join(logdir_root, STARTED_DATESTRING)
  print('Using default logdir: {}'.format(logdir))
  return logdir



def main(unused_args):
  if args.logdir is None:
    raise ValueError('Please specify the dir to the checkpoint')

  speaker_list = txt2list(args.speaker_list)

  arch = tf.gfile.Glob(os.path.join(args.logdir, 'arch*.json'))[0]
  arch = json2dict(arch)

  net = VQVAE(arch)

  # they start roughly at the same position but end very differently (3 is longest)
  filenames = [
    'dataset/VCTK/tfr/p227/p227_363.tfr',
    'dataset/VCTK/tfr/p240/p240_341.tfr',
    'dataset/VCTK/tfr/p243/p243_359.tfr',
    'dataset/VCTK/tfr/p231/p231_430.tfr']
  data = ByteWavWholeReader(speaker_list, filenames)

  X = tf.placeholder(dtype=tf.int64, shape=[None, None])
  Y = tf.placeholder(dtype=tf.int64, shape=[None,])
  ZH = net.encode(X, args.mode)
  XH = net.generate(X, ZH, Y)
  XWAV = mu_law_decode(X)
  XBIN = tf.contrib.ffmpeg.encode_audio(XWAV, 'wav', arch['fs'])


  ema = tf.train.ExponentialMovingAverage(decay=0.995)
  trg_vars = {ema.average_name(v): v for v in tf.trainable_variables()}
  saver = tf.train.Saver(trg_vars)

  logdir = get_default_logdir(args.logdir)
  tf.gfile.MkDir(logdir)


  sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(allow_growth=True))
  with tf.Session(config=sess_config) as sess:
    sess.run(tf.tables_initializer())
    sess.run(data.iterator.initializer)

    results = []
    for _ in filenames:
      result = sess.run({'x': data.x, 'y': data.y})
      results.append(result)
    # results1 = sess.run({'x': data.x, 'y': data.y})
    # results2 = sess.run({'x': data.x, 'y': data.y})
    
    length_input = net.n_padding() + 1  # same as padding + 1

    ini = 15149 - length_input
    end = 42285
    # x_source1 = results1['x'][:, ini: end]
    # x_source2 = results2['x'][:, ini: end]
    for i in range(len(results)):
      x = results[i]['x']
      if x.shape[-1] < end:
        x = np.concatenate([x, x[0,0] + np.zeros([1, end - x.shape[-1]])], -1)
      results[i]['x'] = x[:, ini: end]
    
    # from pdb import set_trace
    # set_trace()
    x_source = np.concatenate(
      [results[0]['x'],
       results[0]['x'],
       results[1]['x'],
       results[1]['x'],
       results[2]['x'],
       results[2]['x'],
       results[3]['x'],
       results[3]['x']],
      0)

    B = x_source.shape[0]

    y_input = np.concatenate(
        [results[0]['y'],
         results[3]['y'],
         results[1]['y'],
         results[0]['y'],
         results[2]['y'],
         results[3]['y'],
         results[3]['y'],
         results[0]['y']],
      0)

    length_target = x_source.shape[1] - length_input

    while True:
      sess.run(tf.global_variables_initializer())
      load(saver, sess, args.logdir, ckpt=args.ckpt)

      z_blend = sess.run(ZH, feed_dict={X: x_source})
      x_input = x_source[:, :length_input]

      z_input = z_blend[:, :length_input, :]

      # Generate
      try:
        x_gen = np.zeros([B, length_target], dtype=np.int64) #+ results['x'][0, 0]
        for i in range(length_target):
          xh = sess.run(XH, feed_dict={X: x_input, ZH: z_input, Y: y_input})
          z_input = z_blend[:, i + 1: i + 1 + length_input, :]
          x_input[:, :-1] = x_input[:, 1:]
          x_input[:, -1] = xh[:, -1]
          x_gen[:, i] = xh[:, -1]
          print('\rGenerating {:5d}/{:5d}... x={:3d}'.format(
            i + 1, length_target, xh[0, -1]), end='', flush=True)
      except KeyboardInterrupt:
        print("Interrupted by the user.")
      finally:
        print()
        x_wav = sess.run(XWAV, feed_dict={X: x_gen})
        for i in range(x_wav.shape[0]):
          x_1ch = np.expand_dims(x_gen[i], -1)
          x_bin = sess.run(XBIN, feed_dict={X: x_1ch})
          with open(os.path.join(logdir, 'testwav-{}.wav'.format(i)), 'wb') as fp:
            fp.write(x_bin)

      # For periodic gen.
      if args.period > 0:
        try:
          print('Sleep for a while')
          sleep(args.period * 60)
          logdir = get_default_logdir(args.logdir)
          tf.gfile.MkDir(logdir)
        except KeyboardInterrupt:
          print('Stop periodic gen.')
          break
        finally:
          print('all finished')
      else:
        break


if __name__ == '__main__':
  tf.app.run()
