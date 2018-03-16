import os
import tensorflow as tf

from util.audio import mu_law_decode
from util.wrapper import txt2list
from dataloader.vctk import ByteWavWholeReader


flags = tf.app.flags
flags.DEFINE_string('output_dir', None, 'output dir')
flags.DEFINE_string('speaker_list', None, 'global control list')
flags.DEFINE_string('file_pattern', None, 'file pattern')
args = flags.FLAGS


def main(_):
  tf.gfile.MkDir(args.output_dir)

  data = ByteWavWholeReader(
    speaker_list=txt2list(args.speaker_list),
    filenames=tf.gfile.Glob(args.file_pattern),
    num_epoch=1)

  XNOM = data.f[0]
  XWAV = tf.expand_dims(mu_law_decode(data.x[0, :]), -1)
  XBIN = tf.contrib.ffmpeg.encode_audio(XWAV, 'wav', 16000)

  sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    gpu_options=tf.GPUOptions(allow_growth=True))
  with tf.Session(config=sess_config) as sess:
    sess.run(tf.tables_initializer())
    sess.run(data.iterator.initializer)
    csv = open('vctk.csv', 'w')
    counter = 1
    while True:
      try:
        fetch = {'xbin': XBIN, 'xwav': XWAV, 'wav_name': XNOM}  
        result = sess.run(fetch)
        wav_name = result['wav_name'].decode('utf8')
        print('\rFile {:05d}: Processing {}'.format(counter, wav_name), end='')
        csv.write('{}, {:d}\n'.format(wav_name, len(result['xwav'])))
        filename = os.path.join(args.output_dir, wav_name) + '.wav'
        with open(filename, 'wb') as fp:
          fp.write(result['xbin'])
        counter += 1
      except tf.errors.OutOfRangeError:
        print('\nEpoch complete')
        break
    print()
    csv.close()


if __name__ == '__main__':
  tf.app.run()
