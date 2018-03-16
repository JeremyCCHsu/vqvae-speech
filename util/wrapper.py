import json
import os
import sys
from datetime import datetime

import tensorflow as tf


def txt2list(filename):
  with open(filename, encoding='utf8') as fp:
    lines = fp.readlines()
  return [n.strip() for n in lines]


def json2dict(filename):
  ''' Read an architecture from a *.json file '''
  with open(filename) as fp:
    arch = json.load(fp)
  return arch


def copy_arch_file(arch, logdir):
  tf.gfile.MakeDirs(logdir)
  _, iFilename = os.path.split(arch)
  oFilename = os.path.join(logdir, iFilename)
  tf.gfile.Copy(arch, oFilename)


def save(saver, sess, logdir, step):
  ''' Save a model to logdir/model.ckpt-[step] '''
  model_name = 'model.ckpt'
  checkpoint_path = os.path.join(logdir, model_name)
  print('Storing checkpoint to {} ...'.format(logdir), end="")
  sys.stdout.flush()

  if not os.path.exists(logdir):
    os.makedirs(logdir)

  saver.save(sess, checkpoint_path, global_step=step)
  print(' Done.')


def load(saver, sess, logdir, ckpt=None):
  '''
  Try to load model form a dir (search for the newest checkpoint)
  '''
  print('Trying to restore checkpoints from {} ...'.format(logdir))
  if ckpt:
    ckpt = os.path.join(logdir, ckpt)
    global_step = int(
      ckpt
      .split('/')[-1]
      .split('-')[-1])
    print('  Global step: {}'.format(global_step))
    print('  Restoring...', end="")
    saver.restore(sess, ckpt)
    return global_step
  else:
    ckpt = tf.train.get_checkpoint_state(logdir)
    if ckpt:
      ckpt_path = ckpt.model_checkpoint_path
      print('  Checkpoint found: {}'.format(ckpt_path))
      print('  Restoring...')
      saver.restore(sess, ckpt_path)
      return None
    else:
      print('No checkpoint found')
      return None


def get_default_logdir(logdir_root, msg=''):
  STARTED_DATESTRING = datetime.now().strftime('%0m%0d-%0H%0M-%0S-%Y')
  logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING + msg)
  print('Using default logdir: {}'.format(logdir))        
  return logdir


def validate_log_dirs(args):
  ''' Create a default log dir (if necessary) '''
  if args.logdir and args.restore_from:
    raise ValueError(
      'You can only specify one of the following: ' +
      '--logdir and --restore_from')

  if args.logdir and args.log_root:
    raise ValueError(
      'You can only specify either --logdir or --logdir_root')

  if args.logdir_root is None:
    logdir_root = 'logdir'

  if args.logdir is None:
    if hasattr(args, 'msg'):
      msg = args.msg
    logdir = get_default_logdir(logdir_root, msg)

  # Note: `logdir` and `restore_from` are exclusive
  if args.restore_from is None:
    restore_from = None
  else:
    restore_from = args.restore_from

  return {
    'logdir': logdir,
    'logdir_root': logdir_root,
    'restore_from': restore_from,
  }

