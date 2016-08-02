from __future__ import division

from graph_builder import GraphBuilder

import numpy as np
import tensorflow as tf

from ops import Conv2D, MaxPool
from batch_norm import BatchNorm


class CNN(GraphBuilder):

    def __init__(self, f, ch, pool, act, use_bn, wd=None, use_stride=False,
                 scope='cnn', trainable=True, initialization='msra'):
        """Add CNN. N = number of layers.

        Args:
            f: filter size, list of N int
            ch: number of channels, list of (N + 1) int
            pool: pooling ratio, list of N int
            act: activation function, list of N function
            use_bn: whether to use batch normalization, list of N bool
            wd: weight decay
        """
        self.filter_size = f
        self.channels = ch
        self.pool = pool
        self.act = act
        self.use_bn = use_bn
        self.wd = wd
        self.scope = scope
        self.trainable = trainable

        self.nlayers = len(f)
        self.w = [None] * self.nlayers
        self.b = [None] * self.nlayers
        self.batch_norm = []
        self.num_copies = 0
        if initialization == 'msra':
            self.compute_std = lambda s: np.sqrt(2 / s[0] * s[1] * s[3])
        else:
            self.compute_std = lambda s: 0.01

        super(CNN, self).__init__()

        self.log.info('CNN: {}'.format(scope))
        self.log.info('Channels: {}'.format(ch))
        self.log.info('Activation: {}'.format(act))
        self.log.info('Pool: {}'.format(pool))
        self.log.info('BN: {}'.format(use_bn))
        pass

    def init_var(self):
        """Initialize variables."""
        f = self.filter_size
        ch = self.channels
        wd = self.wd
        trainable = self.trainable
        with tf.variable_scope(self.scope):
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    self.w[ii] = self.declare_var(
                        [f[ii], f[ii], ch[ii], ch[ii + 1]],
                        name='w', wd=wd,
                        trainable=trainable,
                        stddev=self.compute_std(
                            [f[ii], f[ii], ch[ii], ch[ii + 1]])
                    )
                    self.b[ii] = self.declare_var(
                        [ch[ii + 1]], name='b',
                        trainable=trainable,
                        stddev=0
                    )
                    self.log.info('Filter: {}, Trainable: {}'.format(
                        [f[ii], f[ii], ch[ii], ch[ii + 1]], trainable))
                    pass
                pass
            pass
        pass

    def get_layer(self, n):
        """Get a layer."""
        return self.hidden_layers[n]

    def build(self, inp):
        """Run CNN on an input.

        Args:
            input: input image, [B, H, W, D]
            phase_train: phase train, bool
        """
        self.lazy_init_var()
        x = inp['input']
        phase_train = inp['phase_train']
        h = [None] * self.nlayers
        self.batch_norm.append([None] * self.nlayers)
        with tf.variable_scope(self.scope):
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    out_ch = self.channels[ii + 1]

                    if ii == 0:
                        prev_inp = x
                    else:
                        prev_inp = h[ii - 1]

                    h[ii] = Conv2D(self.w[ii])(prev_inp) + self.b[ii]

                    if self.use_bn[ii]:
                        self.batch_norm[self.num_copies][
                            ii] = BatchNorm(out_ch)
                        h[ii] = self.batch_norm[self.num_copies][ii](
                            {'input': h[ii], 'phase_train': phase_train})

                    if self.act[ii] is not None:
                        h[ii] = self.act[ii](h[ii])

                    if self.pool[ii] > 1:
                        h[ii] = MaxPool(self.pool[ii])(h[ii])
                    pass
                pass
            pass
        self.num_copies += 1
        self.hidden_layers = h
        return h[-1]

    def get_save_var_dict(self):
        results = {}
        for ii in xrange(self.nlayers):
            prefix = 'layer_{}/'.format(ii)
            results[prefix + 'w'] = self.w[ii]
            results[prefix + 'b'] = self.b[ii]
        for cc in xrange(self.num_copies):
            for ii in xrange(self.nlayers):
                prefix = 'layer_{}/'.format(ii)
                if len(self.batch_norm) == 1:
                    bn_name = 'bn'
                else:
                    bn_name = 'bn_{}'.format(cc)
                bn = self.batch_norm[cc][ii]
                if bn is not None:
                    self.add_prefix_to(
                        prefix + bn_name, bn.get_save_var_dict(), results)
        return results
    pass
