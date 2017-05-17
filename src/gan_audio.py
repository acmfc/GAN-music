import argparse
import json

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os

from gan_audio_reader import GanAudioReader, calculate_receptive_field
from wavenet.model import WaveNetModel
from wavenet.ops import *


def xavier_init(size):
  in_dim = size[0]
  xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
  return tf.random_normal(shape=size, stddev=xavier_stddev)


def sample_Z(m, n):
  return np.random.normal(size=(m, n))


def make_layer(in_dim, out_dim):
  W = tf.Variable(xavier_init([in_dim, out_dim]))
  b = tf.Variable(tf.zeros(shape=[out_dim]))
  return W, b


# TODO Add support for generating arbitrary length samples by generating
# multiple batches, each conditioned on all previous samples.
class RecurrentGenerator:
  '''Generator in the style of GRAN as presented in Generating Images with
  Recurrent Adversarial Networks.'''
  def __init__(self, total_sample_size, z_dim, hidden_dim, time_steps=5):
    self.loss = None

    self.encoder_W1, self.encoder_b1 = make_layer(total_sample_size, hidden_dim)
    self.encoder_W2, self.encoder_b2 = make_layer(hidden_dim, hidden_dim)
    self.encoder_W3, self.encoder_b3 = make_layer(hidden_dim, hidden_dim)

    self.decoder_Z_W, self.decoder_Z_b = make_layer(z_dim, hidden_dim)
    self.decoder_W1, self.decoder_b1 = make_layer(2 * hidden_dim, 2 * hidden_dim)
    self.decoder_W2, self.decoder_b2 = make_layer(2 * hidden_dim, 2 * hidden_dim)
    self.decoder_W3, self.decoder_b3 = make_layer(2 * hidden_dim, total_sample_size)

    self.theta = [
        self.encoder_W1, self.encoder_b1,
        self.encoder_W2, self.encoder_b2,
        self.encoder_W3, self.encoder_b3,
        self.decoder_Z_W, self.decoder_Z_b,
        self.decoder_W1, self.decoder_b1,
        self.decoder_W2, self.decoder_b2,
        self.decoder_W3, self.decoder_b3
        ]


    self.Z = tf.placeholder(tf.float32, shape=[None, z_dim])
    # canvases is a list of (Ct, H_Ct) with length time_steps.
    canvases = [[None, None] for _ in range(time_steps)]
    canvases[0][1] = tf.zeros((1, hidden_dim))
    for i in range(time_steps - 1):
      canvases[i][0] = self.decoder(canvases[i][1])
      canvases[i + 1][1] = self.encoder(canvases[i][0])
    self.sample = tf.nn.sigmoid(
        tf.add_n([ct for ct, _ in canvases[:time_steps - 1]]))


  def decoder(self, H_Ct):
    h0 = tf.nn.tanh(tf.matmul(self.Z, self.decoder_Z_W) + self.decoder_Z_b)
    r = tf.concat([h0, H_Ct], 1)
    h1 = tf.nn.relu(tf.matmul(r, self.decoder_W1) + self.decoder_b1)
    h2 = tf.nn.relu(tf.matmul(h1, self.decoder_W2) + self.decoder_b2)
    return tf.tanh(tf.matmul(h2, self.decoder_W3) + self.decoder_b3)


  def encoder(self, Ct):
    h0 = tf.nn.relu(tf.matmul(Ct, self.encoder_W1) + self.encoder_b1)
    h1 = tf.nn.relu(tf.matmul(h0, self.encoder_W2) + self.encoder_b2)
    return tf.tanh(tf.matmul(h1, self.encoder_W3) + self.encoder_b3)


  def set_discriminator(self, D):
    self.loss = -tf.reduce_mean(D.fake)


class Discriminator:
  def __init__(self, X, G_sample, total_sample_size, hidden_dim):
    D_W1, D_b1 = make_layer(total_sample_size, hidden_dim)
    D_W2, D_b2 = make_layer(hidden_dim, hidden_dim)
    D_W3, D_b3 = make_layer(hidden_dim, 1)
    self.theta = [D_W1, D_W2, D_W3, D_b1, D_b2, D_b3]

    def make_network(X):
      D_h1 = tf.nn.relu(tf.matmul(X, D_W1) + D_b1)
      D_h2 = tf.nn.relu(tf.matmul(D_h1, D_W2) + D_b2)
      return tf.matmul(D_h2, D_W3) + D_b3

    self.real = make_network(X)
    self.fake = make_network(G_sample)
    self.loss = tf.reduce_mean(self.real) - tf.reduce_mean(self.fake)
    self.clip = [v.assign(tf.clip_by_value(v, -0.01, 0.01)) for v in self.theta]


def main(args):
  with open(args.wavenet_params, 'r') as f:
    receptive_field = calculate_receptive_field(json.load(f))
  total_sample_size = receptive_field + args.sample_size

  HIDDEN_DIM = 1024
  Z_DIM = 100

  X = tf.placeholder(tf.float32, shape=[None, total_sample_size])

  G = RecurrentGenerator(total_sample_size, Z_DIM, HIDDEN_DIM)
  D = Discriminator(X, G.sample, total_sample_size, HIDDEN_DIM)
  G.set_discriminator(D)

  print('Creating D_solver')
  D_solver = tf.train.RMSPropOptimizer(
      learning_rate=args.learning_rate).minimize(-D.loss, var_list=D.theta)

  print('Creating G_solver')
  G_solver = tf.train.RMSPropOptimizer(
      learning_rate=args.learning_rate).minimize(G.loss, var_list=G.theta)

  sess = tf.Session()
  print('Initializing variables')
  sess.run(tf.global_variables_initializer())

  audio_reader = GanAudioReader(args, sess, receptive_field)

  G_decode = mu_law_decode(tf.to_int32(G.sample * args.quantization_channels),
      args.quantization_channels)

  for it in range(args.iters):
    for _ in range(5):
      X_mb = audio_reader.next_audio_batch()
      if len(X_mb) < total_sample_size:
        X_mb = np.pad(X_mb, ((0, total_sample_size - len(X_mb))), 'constant',
            constant_values=0.)
      X_mb = X_mb.reshape([1, total_sample_size])

      _, D_loss_curr, _ = sess.run([D_solver, D.loss, D.clip],
          feed_dict={X: X_mb, G.Z: sample_Z(args.batch_size, Z_DIM)})

    _, G_loss_curr = sess.run([G_solver, G.loss],
        feed_dict={G.Z: sample_Z(args.batch_size, Z_DIM)})

    if it % 10 == 0:
      print('Iter: {}\nD loss: {:.4}\nG loss: {:.4}\n'.format(
        it, D_loss_curr, G_loss_curr))
      if it % 50 == 0:
        samples, decoded = sess.run([G.sample, G_decode],
            feed_dict={G.Z: sample_Z(args.batch_size, Z_DIM)})
        with open('output_{}'.format(it), 'w') as f:
          f.write(','.join(str(sample) for sample in samples[0]))
          f.write('\n')
          f.write(','.join(str(sample) for sample in decoded[0]))

  audio_reader.done()


LEARNING_RATE = 5e-5
WAVENET_PARAMS = '../config/wavenet_params.json'
# STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.01
# EPSILON = 0.001
MOMENTUM = 0.9
# MAX_TO_KEEP = 5
# METADATA = False


def parse_args():
  parser = argparse.ArgumentParser(description='Quick hack together of audio_reader + gan_tensorflow')
  parser.add_argument('audio_dir',
    help='Directory containing input WAV files')
  parser.add_argument('--iters', type=int, default=1000000)
  parser.add_argument('--sample_rate', type=int, default=16000)
  parser.add_argument('--file_seconds', help='Number of audio seconds to use from each input file',
    type=int, default=29)
  parser.add_argument('--quantization_channels', type=int, default=256)
  parser.add_argument('--gc_enabled', action='store_true')
  parser.add_argument('--batch_size', type=int, default=1)
  parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
    help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
  parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
    help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
  parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
    help='Concatenate and cut audio samples to this many '
         'samples. Default: ' + str(SAMPLE_SIZE) + '.')
  parser.add_argument('--l2_regularization_strength', type=float,
    default=L2_REGULARIZATION_STRENGTH,
    help='Coefficient in the L2 regularization. '
         'Default: False')
  parser.add_argument('--silence_threshold', type=float,
    default=SILENCE_THRESHOLD,
    help='Volume threshold below which to trim the start '
         'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')

  args = parser.parse_args()

  args.samples = args.file_seconds * args.sample_rate
  print('Extracting %s samples per file' % args.samples)

  return args


if __name__ == '__main__':
  args = parse_args()
  main(args)
