from os.path import join, split, splitext

import tensorflow as tf

from util.audio import mu_law_encode
from util.wrapper import txt2list
from dataloader.vctk import make_mu_law_speaker_length

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('fs', 16000, 'sampling freq')
tf.app.flags.DEFINE_string('file_pattern', None, '')
tf.app.flags.DEFINE_string('output_dir', None, '')
tf.app.flags.DEFINE_string('speaker_list', None, '')
tf.app.flags.DEFINE_string('ext', 'wav',
  'file extension: wav, mp3, mp4, ogg are supported.')


def read_text(filename):
  ''' dedicated to VCTK '''
  filename = filename.replace('wav48', 'txt')
  filename = filename.replace('.wav', '.txt')
  try:
    with open(filename, 'r', encoding='utf8') as fp:
      lines = fp.readlines()
    lines = ''.join(lines)
  except FileNotFoundError:
    print('[WARNING] text not found: {}'.format(filename))
    lines = ''
  finally:
    pass
  return lines


def main(unused_args):
  '''
  NOTE: the directory structure must be [args.dir_to_wav]/[Set]/[speakers]
  '''
  if not args.output_dir:
    raise ValueError('`output_dir` (output dir) should be specified')

  print('[WARNING] Protobuf is super slow (~7 examples per sec). \n'
    'This could take 2 hours or more.')

  reader = tf.WholeFileReader()
  files = tf.gfile.Glob(args.file_pattern)
  filename_queue = tf.train.string_input_producer(
    files,
    num_epochs=1,
    shuffle=False)

  key, val = reader.read(filename_queue)
  wav = tf.contrib.ffmpeg.decode_audio(val, args.ext, args.fs, 1)
  wav = tf.reshape(wav, [-1,])
  mulaw = mu_law_encode(wav)

  for s in txt2list(args.speaker_list):
    tf.gfile.MakeDirs(join(args.output_dir, s))

  counter = 1
  N = len(files)
  with tf.train.MonitoredSession() as sess:
    while not sess.should_stop():

      filename, x_int = sess.run([key, mulaw])
      filename = filename.decode('utf8')

      text = read_text(filename)

      b, _ = splitext(filename)
      _, b = split(b)

      s = b.split('_')[0]

      ex = make_mu_law_speaker_length(x_int, s, text, b)

      fp = tf.python_io.TFRecordWriter(
        join(args.output_dir, s, '{}.tfr'.format(b)))
      fp.write(ex.SerializeToString())
      fp.close()

      print('\rFile {:5d}/{:5d}: {}'.format(counter, N, b), end='')      
      counter += 1

  print()


if __name__ == '__main__':
  tf.app.run()
