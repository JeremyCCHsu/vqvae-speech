# encodings=utf8

import tensorflow as tf
import numpy as np

from util.audio import mu_law_decode, spectrogram
from util.train import accuracy_of_minibatch, visualize_embeddings
from util.wrapper import load


WEIGHT_DECAY = 1e-10


class VQVAE(object):
  '''
  Vector-Quantization Variational Auto-encoder
  '''
  def __init__(self, arch):
    self.arch = arch
    with tf.variable_scope('Embedding'):
      if arch['dim_symbol_emb'] == 1:
        self.x_emb = tf.reshape(
          tf.range(arch['num_symbol'], dtype=tf.float32),
          [-1, 1]
        )
        self.x_emb = tf.get_variable(
          name='SymbolEmb',
          initializer=self.x_emb / (arch['num_symbol'] / 2) - 1,
          regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
          trainable=False,
        )
      else:
        self.x_emb = tf.get_variable(
          name='SymbolEmb',
          shape=[arch['num_symbol'], arch['dim_symbol_emb']],
          initializer=tf.random_normal_initializer(stddev=1.),
          regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
        )

      self.z_emb = tf.get_variable(
        name='ExemplarEmb',
        shape=[arch['num_exemplar'], arch['dim_exemplar']],
        initializer=tf.random_normal_initializer(stddev=4.5e-3),
        regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      )

      self.y_emb = tf.get_variable(
        name='SpeakerEmb',
        shape=[arch['num_speaker'], arch['dim_speaker']],
        regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
        initializer=tf.orthogonal_initializer,
      )

    self.encodings = tf.get_variable(
      name='vizualized_encodings',
      shape=[arch['n_emb_project'] + arch['num_exemplar'],
             arch['dim_exemplar']],
      trainable=False,
    )
    self.encoding_placeholder = tf.placeholder(
      dtype=tf.float32,
      shape=[None, None]
    )

    self._Enc = tf.make_template('Encoder', self._encoder)
    self._Dec = tf.make_template('Decoder', self._wavenet)

  
  def _encoder(self, x):
    '''
    Note that we need a pair of reversal to ensure causality.
    (i.e. no trailing pads)
    `x`: [b, T, c]
    '''
    k_init = self.arch['initial_filter_width']

    b = tf.shape(x)[0]
    o = tf.zeros([b, k_init - 1, self.arch['dim_symbol_emb']])
    x = tf.concat([o, x], 1)

    k_init = self.arch['initial_filter_width']
    x = tf.layers.conv1d(
      inputs=x,
      filters=self.arch['residual_channels'],
      kernel_size=k_init,
      kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      name='initial_filtering',
      kernel_initializer=tf.initializers.variance_scaling(
        scale=1.43,
        distribution='uniform'),
    )
    x = tf.nn.leaky_relu(x, 2e-2)

    x = tf.reverse(x, [1])  # paired op to enforce causality
    for i in range(self.arch['n_downsample_stack']):
      conv = tf.layers.conv1d(
        inputs=x,
        filters=(i + 1) * self.arch['encoder']['filters'],
        kernel_size=self.arch['encoder']['kernels'],
        strides=2,
        padding='same',
        # activation=tf.nn.tanh,
        kernel_initializer=tf.initializers.variance_scaling(
          scale=1.15,
          distribution='uniform'),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      )
      gate = tf.layers.conv1d(
        inputs=x,
        filters=(i + 1) * self.arch['encoder']['filters'],
        kernel_size=self.arch['encoder']['kernels'],
        strides=2,
        padding='same',
        # activation=tf.nn.sigmoid,
        kernel_initializer=tf.initializers.variance_scaling(distribution='uniform'),
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
        bias_initializer=tf.initializers.ones,
      )
      x = tf.nn.tanh(conv) * tf.nn.sigmoid(gate)
    x = tf.reverse(x, [1])  # paired op to enforce causality

    x = tf.layers.conv1d(
      inputs=x,
      filters=self.arch['dim_exemplar'],
      kernel_size=1,
      kernel_initializer=tf.initializers.variance_scaling(
          distribution='uniform'),
      kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
    )
    return x


  # ====================== WaveNet ======================
  def _wavenet(self, x, y=None, h=None):
    ''' WaveNet Decoder
    NOTE: I'm assuming that the input is already padded.
    `x`: wav signal. `float`, [b, T, c]
    `y`: local conditional signal. Required to be padded. `int`, [b, T]
    `h`: global conditional signal. `int`, [b,]
    '''
    k_init = self.arch['initial_filter_width']
    with tf.name_scope('WaveNet'):
      x = tf.layers.conv1d(
        inputs=x,
        filters=self.arch['residual_channels'],
        kernel_size=k_init,
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
        name='initial_filtering',
      )
      if y is not None:
        y = y[:, k_init -1:]

      with tf.name_scope('WavenetBlocks'):
        all_outputs = list()
        for d in self.arch['dilations']:
          x, output, y = self._wavenet_block(x, y=y, h=h, dilation=d)
          all_outputs.append(output)

        T = tf.shape(all_outputs[-1])[1]

        all_outputs = [v[:, -T:, :] for v in all_outputs]  # truncate to valid length
        all_outputs = tf.add_n(all_outputs)

      x = tf.nn.relu(all_outputs)
      x = tf.layers.conv1d(
        inputs=x,
        filters=self.arch['skip_channels'],
        kernel_size=1,
        activation=tf.nn.relu,
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      )
      logits = tf.layers.conv1d(
        inputs=x,
        filters=self.arch['num_symbol'],
        kernel_size=1,
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      )
      return logits


  def _subblock(self, x, y=None, h=None, filters=32, kernels=2, dilation=2, name='Gate'):
    ''' Wavenet uses different net for conv and gate (Eqn 2) '''
    with tf.name_scope(name):
      conv = tf.layers.conv1d(
        x, filters, kernels,
        dilation_rate=dilation,
        kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
      )

      # Global conditional signal
      if h is not None:
        with tf.name_scope('GlobalCondition'):
          h = tf.nn.embedding_lookup(self.y_emb, h)
          h = tf.expand_dims(h, 1)  # [b, c] -> [b, 1, c]
          h_conv = tf.layers.dense(h, filters, use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
          )
          conv += h_conv

      # Local conditional signal
      if y is not None:
        with tf.name_scope('LocalCondition'):
          start_idx = (kernels - 1) * dilation
          y = y[:, start_idx:]
          y_analog = y
          y_conv = tf.layers.dense(y_analog, filters, use_bias=False,
            kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
          )
          conv += y_conv
      return conv, y


  def _wavenet_block(self, x, y=None, h=None, filters=32, kernels=2, dilation=2):
    ''' WaveNet Block
    `x`: wav signal. `float`, [b, T, c]
    `y`: local conditional signal. Required to be padded. `float`, [b, T, d]
    `h`: global conditional signal. `int`, [b,]
    '''
    with tf.name_scope('WaveNet_Block'):
      conv, _ = self._subblock(x, y, h, filters, kernels, dilation, name='Conv')
      gate, y = self._subblock(x, y, h, filters, kernels, dilation, name='Gate')

      gated = tf.nn.tanh(conv) * tf.nn.sigmoid(gate)

      with tf.name_scope('ResidualConnection'):
        res = tf.layers.conv1d(
          inputs=gated,
          filters=self.arch['residual_channels'],
          kernel_size=1,
          kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
        )
        start_idx = (kernels - 1) * dilation
        x = x[:, start_idx:]
        res = x + res

      with tf.name_scope('SkipConnection'):
        skip = tf.layers.conv1d(
          inputs=gated,
          filters=self.arch['skip_channels'],
          kernel_size=1,
          kernel_regularizer=tf.keras.regularizers.l2(WEIGHT_DECAY),
        )

      return res, skip, y
  # ====================== WaveNet ======================
  

  # ====================== Utility functions for `loss`` ======================
  def _C(self, x):
    '''
    Nearest neighbor (to z_emb) search
    `x`: [b, T, c]
    `z_e`: [K, c]
    '''
    with tf.name_scope('Classifier'):
      similarity = tf.tensordot(x, self.z_emb, [[-1], [-1]])
      tf.summary.histogram('x_dot_ze', similarity)

      z2 = tf.reduce_sum(tf.square(self.z_emb), axis=-1)  # [K]
      tf.summary.histogram('z_norm_2', z2)
      tf.summary.histogram('z', self.z_emb)

      x2 = tf.reduce_sum(tf.square(x), axis=-1) # [b, T]
      tf.summary.histogram('x_norm_2', x2)
      tf.summary.histogram('x', x)

      dist2 = tf.nn.bias_add(- 2. * similarity, z2) + tf.expand_dims(x2, axis=-1)  # x2 -2xz + z2

      u, v = tf.nn.moments(x, axes=[-1])
      tf.summary.histogram('x_mu', u)
      tf.summary.histogram('x_var', v)

      u, v = tf.nn.moments(self.z_emb, axes=[-1])
      tf.summary.histogram('z_mu', u)
      tf.summary.histogram('z_var', v)

      z_ids = tf.argmin(dist2, axis=-1)
      tf.summary.histogram('z_ids', z_ids)
      return z_ids


  def _D2A(self, x):
    return tf.nn.embedding_lookup(self.x_emb, x)


  def n_padding(self):
    ''' num of paddings '''
    filter_width = self.arch['filter_width']
    dilations = self.arch['dilations']
    initial_filter_width = self.arch['initial_filter_width']

    receptive_field = (filter_width - 1) * sum(dilations) + 1
    receptive_field += initial_filter_width - 1
    return receptive_field - 1


  def _uptile(self, x):
    '''Upsampling by replicating the values
    `x`: [b, t, c]
    '''
    shape = tf.shape(x)
    n_replica = 2 ** self.arch['n_downsample_stack']
    x = tf.tile(x, [1, 1, n_replica])
    x = tf.reshape(x, [shape[0], shape[1] * n_replica, self.arch['dim_exemplar']])
    return x


  def loss(self, x, y):
    '''
    `loss` needs rewriting because the whole procedure changes (we're using exe now).
    Need to add VQ part into `loss`
    `x`: [b, T] (unpadded, `int`)
    `y`: [b,] (`int`)
    '''
    tf.summary.histogram('y', y)

    T = tf.shape(x)[1]
    x_analog = self._D2A(x)
    z_enc = self._Enc(x_analog)

    # Passing tiled embeddings might seem inefficient, but it's necessary for TF.
    z_ids = self._C(z_enc)
    z_exe = tf.nn.embedding_lookup(self.z_emb, z_ids)
    z_exe_up = self._uptile(z_exe)
    z_exe_up = z_exe_up[:, -T:]
    x_ar = self._Dec(x_analog, z_exe_up, y)

    with tf.name_scope('Loss'):
      P = self.n_padding()
      x_answer_ar = x[:, P + 1:]
      x_pred_ar_logits = x_ar[:, :-1]  # no answer for the last prediction

      xh = tf.argmax(x_pred_ar_logits, axis=-1)

      loss_ar = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=x_pred_ar_logits,
          labels=x_answer_ar,
        )
      )
      accuracy_of_minibatch(x_answer_ar, xh)


      # loss = self._Wasserstein_objective(z_enc)
      loss = {}
      loss['reconst'] = loss_ar

      loss['vq'] = tf.reduce_mean(tf.reduce_sum(tf.square(z_enc - z_exe), -1))
      tf.summary.scalar('vq', loss['vq'])

      loss_reg = tf.reduce_sum(tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
      )
      tf.summary.scalar('loss_reg', loss_reg)
      loss['reg'] = loss_reg

      # tf.summary.audio('x', mu_law_decode(x), self.arch['fs'])
      # tf.summary.audio('xh', mu_law_decode(xh), self.arch['fs'])
      tf.summary.scalar('xent_ar', loss_ar)
      tf.summary.histogram('log_z_enc_norm', tf.log(tf.norm(z_enc, axis=-1)))
      # TODO: only applicable to audio input with dim_symbol_emb=1.
      # spectrogram(x_analog, 2 ** self.arch['n_downsample_stack'])

      loss.update({'z_enc': z_enc, 'z_exe': z_exe})
      return loss


  def _optimize(self, loss):
    global_step = tf.get_variable(
      name='global_step',
      shape=[],
      dtype=tf.int64,
      initializer=tf.constant_initializer(0),
      trainable=False,
    )
    hyperp = self.arch['training']
    optimizer = tf.train.AdamOptimizer(
      hyperp['lr'], hyperp['beta1'], hyperp['beta2'])

    all_vars = tf.trainable_variables()
    g_vars = [v for v in all_vars if 'Decoder' in v.name or 'SpeakerEmb' in v.name] # NOTE: Symbol is ignored
    e_vars = [v for v in all_vars if 'Encoder' in v.name]
    z_vars = [v for v in all_vars if 'ExemplarEmb' in v.name]

    # ========== Straight-Through ============================
    beta = .25
    l_Exe = loss['vq'] + loss['reg']
    l_Dec = loss['reconst'] + loss['reg']
    l_commit = beta * loss['vq']

    z_enc, z_exe = loss['z_enc'], loss['z_exe']
    grad_from_st = tf.gradients(
      grad_ys=tf.gradients(ys=l_Dec, xs=z_exe),
      ys=z_enc,
      xs=e_vars,
    )
    grad_from_commit = tf.gradients(ys=l_commit, xs=e_vars)
    e_grad = [u + v for u, v in zip(grad_from_st, grad_from_commit)]
    e_grad_vars = [(g, v) for g, v in zip(e_grad, e_vars)]
    # =======================================================

    opt_e = optimizer.apply_gradients(e_grad_vars)
    opt_z = optimizer.minimize(l_Exe, var_list=z_vars)
    opt_g = optimizer.minimize(l_Dec, var_list=g_vars, global_step=global_step)

    ema = tf.train.ExponentialMovingAverage(decay=0.995)
    maintain_averages_op = ema.apply(all_vars)

    with tf.control_dependencies([opt_e, opt_g, opt_z]):
      training_op = tf.group(maintain_averages_op)

    return {'trn': training_op, 'ema': ema}


  def _make_dummy_tsv(self):
    path = 'dummy.tsv'
    with open(path, 'w') as fp:
      for i in range(self.arch['n_emb_project']):
        fp.write('x{:04d}\n'.format(i))
      for i in range(self.arch['num_exemplar']):
        fp.write('c{:03}\n'.format(i))
    return path


  def train(self, data):
    hyperp = self.arch['training']

    loss = self.loss(data.x, data.y)
    opt = self._optimize(loss)

    Z = self._Enc(self._D2A(data.x))
    update_encoding = tf.assign(self.encodings, self.encoding_placeholder)


    K, D = self.arch['num_exemplar'], self.arch['dim_exemplar']
    z_emp = tf.placeholder(tf.float32, [K, D], 'z_emp')
    init_z_emb = tf.assign(self.z_emb, z_emp)

    ema = opt['ema']
    trg_vars = {ema.average_name(v): v for v in tf.trainable_variables()}

    saver = tf.train.Saver(trg_vars)

    sess_config = tf.ConfigProto(
      allow_soft_placement=True,
      gpu_options=tf.GPUOptions(allow_growth=True)
    )
    scaffold = tf.train.Scaffold(
      local_init_op=tf.group(
        tf.local_variables_initializer(),
        data.iterator.initializer,
        tf.tables_initializer()
      )
    )
    with tf.train.MonitoredTrainingSession(
      scaffold=scaffold,
      checkpoint_dir=self.arch['logdir'],
      save_checkpoint_secs=360,
      save_summaries_secs=120,
      config=sess_config,
    ) as sess:
      dummy_path = self._make_dummy_tsv()
      visualize_embeddings(
        logdir=self.arch['logdir'],
        var_list=[self.y_emb, self.encodings],
        tsv_list=['etc/speakers_label.tsv', dummy_path],
      )

      if self.arch['restore_from']:
        load(saver, sess, self.arch['restore_from'], ckpt=self.arch['ckpt'])

      # ========== Initialize exemplars with Enc output  ==========
      multiplier = 100
      exe = np.zeros([0, D])
      while exe.shape[0] < K * multiplier:
        z = sess.run(Z)
        exe = np.concatenate([exe, np.reshape(z, [-1, D])], 0)
      np.unique(exe, axis=0)
      np.random.shuffle(exe)
      sess.run(init_z_emb, feed_dict={z_emp: exe[:K, :]})
      

      # ========== Main training loop ==========
      maxIter = hyperp['maxIter']
      for it in range(maxIter):
        sess.run(opt['trn'])
        if it % hyperp['refresh_freq'] == 1:
          self._get_and_update_encodings(sess, Z, data.y, update_encoding)
          fetches = {'l': loss['reconst']}
          results = sess.run(fetches)
          print('\rIter {:5d}: loss = {:.4e}'.format(it, results['l']), end='')
      print()


  def _get_and_update_encodings(self, sess, z, Y, update_op):
    N = self.arch['n_emb_project']
    n_enc_per_batch = self.arch['T'] \
      // (2 ** self.arch["n_downsample_stack"]) \
      * self.arch['training']['batch_size']
    n = N // n_enc_per_batch + 1
    z_encs = []
    for _ in range(n):
      z_enc, y = sess.run([z, Y])
      z_encs.append(z_enc)
      y = np.expand_dims(y, -1)
      y = np.tile(y, [1, z_enc.shape[1]])

    z_encs = np.concatenate(z_encs, 0)  # [b * m, t, c]

    z_encs = np.reshape(z_encs, [-1, z_encs.shape[-1]])
    z_encs = z_encs[:N]

    z_exe = sess.run(self.z_emb)
    z_encs = np.concatenate([z_encs, z_exe], 0)

    sess.run(update_op, feed_dict={self.encoding_placeholder: z_encs})
    return z_encs
    

  def encode(self, x, mode='exemplar'):
    ''' Encode a wav sequence to a latent (phone) sequence. (ASR-like)
    Input:
      `x`: wav input of shape [b, T]. `int`
    Return:
      `z`: phone sequence of shape [b, T]. `int`
    '''
    print('Encoder output mode: {}'.format(mode))
    with tf.name_scope('EncoderAPI'):
      T = tf.shape(x)[1]
      x_analog = self._D2A(x)
      z_enc = self._Enc(x_analog)

      if mode == 'exemplar':
        z_ids = self._C(z_enc)
        z_exe = tf.nn.embedding_lookup(self.z_emb, z_ids)
        z_exe_up = self._uptile(z_exe)
        return z_exe_up[:, -T:]

      elif mode == 'encoding':
        return self._uptile(z_enc)[:, -T:]
      
      elif mode == 'id':
        z_ids = self._C(z_enc)
        return z_ids
      
      else:
        raise ValueError('Not implemented: {}'.format(mode))


  def generate(self, x=None, y=None, h=None, mode='random'):
    '''
    Output 1 generated sample point.

    Input:
      `x`: wav signal, `int`, [b, T]
      `y`: phone sequence (local conditional signal), `int`, [b, T]
      `h`: speaker id (global conditional signal), `int`, [b,]
      `mode`:
        `random` (default; output randomly from 256 possibilities proportional to logits)
        `greedy`

    Output:
      `xh`: generated sample of next time slot, `int`, [b, 1]

    FIXME: Generation is insanely slow. It could be made much more efficient
           if we don't re-compute the already available values.
           Currently, it takes 10 minutes to generate [4, 16000] samples.
    '''
    print('Decoder output mode: {}'.format(mode))
    with tf.name_scope('IncrementalGeneration'):
      x_analog = self._D2A(x)
      x_next = self._Dec(x_analog, y, h)
      x_next = x_next[:, -1, :]

      # The output mode
      if mode == 'random':
        return tf.multinomial(x_next, 1)  # [b, 1], int64
      elif mode == 'greedy':
        return tf.argmax(x_next, axis=-1)
      else:
        raise NotImplementedError('"Mode: {}" is not implemented'.format(mode))
