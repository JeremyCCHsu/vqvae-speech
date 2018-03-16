import tensorflow as tf
import numpy as np


def mu_law_encode(audio, quantization_channels=256):
  '''Quantizes waveform amplitudes.
  Adapted from https://github.com/ibab/tensorflow-wavenet
  '''
  with tf.name_scope('encode'):
    mu = quantization_channels - 1
    safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
    magnitude = tf.log(1. + mu * safe_audio_abs) / tf.log(1. + mu)
    signal = tf.sign(audio) * magnitude
    return tf.cast((signal + 1) / 2 * mu + 0.5, tf.uint8)

# # Numpy version
# def mu_law_encode(audio, quantization_channels=256):
#   '''Quantizes waveform amplitudes.
#   Adapted from https://github.com/ibab/tensorflow-wavenet
#   '''
#   with tf.name_scope('encode'):
#     mu = quantization_channels - 1
#     safe_audio_abs = np.minimum(np.abs(audio), 1.0)
#     magnitude = np.log1p(mu * safe_audio_abs) / np.log1p(mu)
#     signal = np.sign(audio) * magnitude
#     return ((signal + 1) / 2 * mu + 0.5).astype(np.uint8)


def mu_law_decode(output, quantization_channels=256):
  '''Recovers waveform from quantized values.
  Adapted from https://github.com/ibab/tensorflow-wavenet
  '''
  with tf.name_scope('decode'):
    mu = quantization_channels - 1
    signal = 2 * (tf.to_float(output) / mu) - 1
    magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
    return tf.sign(signal) * magnitude


def gray2jet(x):
  ''' NHWC (channel last) format '''
  def line(x, xa, xb, ya, yb):
    ''' a line determined by two points '''
    return ya + (x - xa) * (yb - ya) / (xb - xa)

  def clip_to_boundary(line1, line2, minval, maxval):
    with tf.name_scope('ClipToBoundary'):
      x = tf.minimum(line1, line2)
      x = tf.minimum(x, maxval)
      x = tf.maximum(x, minval)
      return x

  with tf.name_scope('Gray2Jet'):
    r = clip_to_boundary(
      line(x, .3515, .66, 0., 1.),
      line(x, .8867, 1., 1., .5),
      minval=0.,
      maxval=1.,
    )
    g = clip_to_boundary(
      line(x, .125, .375, 0., 1.),
      line(x, .64, .91, 1., 0.),
      minval=0.,
      maxval=1.,
    )
    b = clip_to_boundary(
      line(x, .0, .1132, 0.5, 1.0),
      line(x, .34, .648, 1., 0.),
      minval=0.,
      maxval=1.,
    )
    return tf.concat([r, g, b], axis=-1)


def spectrogram(x, frame_length, nfft=1024):
  ''' Spectrogram of non-overlapping window '''
  with tf.name_scope('Spectrogram'):
    shape = tf.shape(x)
    b = shape[0]
    D = frame_length
    t = shape[1] // D
    x = tf.reshape(x, [b, t, D])

    window = tf.contrib.signal.hann_window(frame_length)
    window = tf.expand_dims(window, 0)
    window = tf.expand_dims(window, 0) # [1, 1, L]
    x = x * window

    pad = tf.zeros([b, t, nfft - D])
    x = tf.concat([x, pad], -1)
    x = tf.cast(x, tf.complex64)
    X = tf.fft(x)  # TF's API doesn't do padding automatically yet

    X = tf.log(tf.abs(X) + 1e-2)

    X = X[:, :, :nfft//2 + 1]
    X = tf.transpose(X, [0, 2, 1])
    X = tf.reverse(X, [1])
    X = tf.expand_dims(X, -1)

    X = (X - tf.reduce_min(X)) / (tf.reduce_max(X) - tf.reduce_min(X))
    X = gray2jet(X)

    tf.summary.image('spectrogram', X)
    return X


def visualize_wav_prob(x, name):
  with tf.name_scope('VisualizeWavProbability'):
    x = tf.nn.softmax(x)  # [b, T, c]
    x = tf.transpose(x, [0, 2, 1])
    x = tf.expand_dims(x, -1)
    tf.summary.image(name, x)


def visualize_wav(x, n_symbol, name):
  with tf.name_scope('VisualizeWav'):
    x = tf.one_hot(x, n_symbol)  # [b, T,]
    x = tf.transpose(x, [0, 2, 1])
    x = tf.expand_dims(x, -1)
    tf.summary.image(name, x)


def visualize_latent_distr(z_posterior, name):
  '''
  [b, T, c]
  NOTE: feeding [b, h, c] into tf.summary.image results in non-showable TB
  '''
  with tf.name_scope('VisualizeLatentDistr'):
    z_posterior = tf.transpose(z_posterior, [0, 2, 1])
    z_posterior = tf.expand_dims(z_posterior, -1)
    tf.summary.image(name, z_posterior)
    tf.summary.histogram(name, z_posterior)
