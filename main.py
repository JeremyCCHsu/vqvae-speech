import tensorflow as tf

from models.vqvae import VQVAE
from util.wrapper import (copy_arch_file, json2dict, txt2list,
                          validate_log_dirs)
from dataloader.vctk import ByteWavReader

flags = tf.app.flags
flags.DEFINE_string(
  'logdir_root', None, 'root of log dir')
flags.DEFINE_string(
  'logdir', None, 'log dir')
flags.DEFINE_string(
  'restore_from', None, 'restore from dir (not from *.ckpt)')
flags.DEFINE_string(
  'file_pattern',
  'datasets/VCTK/bin/*.tfr',
  ('File patterns of text corpora. '
   'MAKE SURE TAHT YOU USE QUOTATION MARKS '
   'for the whole string that contains *!')
)
flags.DEFINE_string(
  'arch', 'architecture.json', 'network architecture'
)
flags.DEFINE_string(
  'speaker_list',
  './etc/speakers.tsv',
  'List of global control signals (e.g. speaker)'
)
# flags.DEFINE_integer('num_gpus', 1, 'Num of GPUs')
flags.DEFINE_string('msg', '', 'Message to add.')
tf.app.flags.DEFINE_string(
  'ckpt', None, 'model checkpoint name, e.g. model.ckpt-398897')
args = flags.FLAGS


def main(_):
  speaker_list = txt2list(args.speaker_list)
  dirs = validate_log_dirs(args)
  arch = json2dict(args.arch)
  arch.update(dirs)
  arch.update({'ckpt': args.ckpt})
  copy_arch_file(args.arch, arch['logdir'])
  net = VQVAE(arch)
  P = net.n_padding()
  print('Receptive field: {} samples ({:.2f} sec)\n'.format(P, P / arch['fs']))
  data = ByteWavReader(
    speaker_list,
    args.file_pattern,
    T=arch['T'],
    batch_size=arch['training']['batch_size'],
    buffer_size=5000
  )
  net.train(data)

if __name__ == '__main__':
  tf.app.run()
