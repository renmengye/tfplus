from __future__ import division

from graph_builder import GraphBuilder
from batch_norm import BatchNorm
from ops import Conv2D, DilatedConv2D, MaxPool, AvgPool

import numpy as np
import tensorflow as tf

"""
A version of the res net that shares the BN (for the purpose of non-RNN usage).
"""


class ResNetSBN(GraphBuilder):
    """
    Each stage represents a different activation map size.
    Each layer represents a residual connection.
    Each unit represents a non-linear activation inside a layer.

    Args:
        layers: list of int. Number of layers in each stage.
        channels: list of int. Number of channels in each stage.
        strides: list of int. Sub-sample coefficient in each stage.
        bottleneck: bool. Whether do 2 3x3 convs on each layer or 1x1, 3x3,
        1x1 convs on each layer.
        dilation: Whether do subsample on strides (for classification), or do
        dilated convolution on strides (for segmentation)
        scope: main scope name for all variables in this graph.
        shortcut: identity or projection.
    """

    def __init__(self, layers, channels, strides, bottleneck=False,
                 dilation=False, wd=None, scope='res_net',
                 shortcut='projection', initialization='msra',
                 compatible=False, trainable=False):
        super(ResNetSBN, self).__init__()
        self.channels = channels
        self.layers = layers
        self.strides = strides
        self.scope = scope
        self.num_stage = len(layers)
        self.w = [None] * self.num_stage
        self.bn = [None] * self.num_stage
        self.shortcut_w = [None] * self.num_stage
        self.shortcut_bn = [None] * self.num_stage
        # self.b = [None] * self.num_stage
        self.bottleneck = bottleneck
        self.dilation = dilation
        self.shortcut = shortcut
        self.compatible = compatible
        self.trainable = trainable
        self.wd = wd
        if bottleneck:
            self.unit_depth = 3
        else:
            self.unit_depth = 2
        if initialization == 'msra':
            self.compute_std = lambda s: np.sqrt(2 / s[0] / s[1] / s[3])
        else:
            self.compute_std = lambda s: 0.01
        pass
        self.copies = []
        pass

    def compute_in_out(self, kk, ch_in, ch_out):
        if self.bottleneck:
            if kk == 0:
                f_ = 1
                ch_in_ = ch_in
                ch_out_ = int(ch_out / 4)
            elif kk == 1:
                f_ = 3
                ch_in_ = int(ch_out / 4)
                ch_out_ = int(ch_out / 4)
            else:
                f_ = 1
                ch_in_ = int(ch_out / 4)
                ch_out_ = ch_out
        else:
            f_ = 3
            if kk == 0:
                ch_in_ = ch_in
            else:
                ch_in_ = ch_out
            ch_out_ = ch_out
        return f_, ch_in_, ch_out_

    def apply_shortcut(self, prev_inp, ch_in, ch_out, phase_train=None, w=None, bn=None, stride=None):
        if self.shortcut == 'projection':
            if self.dilation:
                prev_inp = DilatedConv2D(w, rate=stride)(prev_inp)
            else:
                prev_inp = Conv2D(w, stride=stride)(prev_inp)
            prev_inp = bn({'input': prev_inp, 'phase_train': phase_train})
        elif self.shortcut == 'identity':
            pad_ch = ch_out - ch_in
            if pad_ch < 0:
                raise Exception('Must use projection when ch_in > ch_out.')
            prev_inp = tf.pad(prev_inp, [[0, 0], [0, 0], [0, 0], [0, pad_ch]])
            if stride > 1:
                prev_inp = AvgPool(stride)(prev_inp)
                raise Exception('DEBUG Unknown')
        self.log.info('After proj shape: {}'.format(
            prev_inp.get_shape()))
        return prev_inp

    def init_var(self):
        with tf.variable_scope(self.scope):
            for ii in xrange(self.num_stage):
                ch_in = self.channels[ii]
                ch_out = self.channels[ii + 1]
                with tf.variable_scope('stage_{}'.format(ii)):
                    self.w[ii] = [None] * self.layers[ii]
                    self.bn[ii] = [None] * self.layers[ii]
                    # self.b[ii] = [None] * self.layers[ii]
                    for jj in xrange(self.layers[ii]):
                        if jj > 0:
                            ch_in = ch_out
                            pass
                        self.w[ii][jj] = [None] * self.unit_depth
                        self.bn[ii][jj] = [None] * self.unit_depth

                        if jj == 0 and (ch_in != ch_out or self.compatible):
                            if self.shortcut == 'projection':
                                with tf.variable_scope('shortcut'):
                                    self.shortcut_w[ii] = self.declare_var(
                                        [1, 1, ch_in, ch_out], wd=self.wd,
                                        name='w',
                                        stddev=self.compute_std(
                                            [1, 1, ch_in, ch_out]),
                                        trainable=self.trainable
                                    )
                                    self.shortcut_bn[ii] = BatchNorm(
                                        ch_out, trainable=self.trainable)
                                pass
                            pass
                        with tf.variable_scope('layer_{}'.format(jj)):
                            for kk in xrange(self.unit_depth):
                                with tf.variable_scope('unit_{}'.format(kk)):
                                    f_, ch_in_, ch_out_ = self.compute_in_out(
                                        kk, ch_in, ch_out)
                                    self.w[ii][jj][kk] = self.declare_var(
                                        [f_, f_, ch_in_, ch_out_], wd=self.wd,
                                        name='w',
                                        stddev=self.compute_std(
                                            [f_, f_, ch_in_, ch_out_]),
                                        trainable=self.trainable
                                    )
                                    if self.compatible:
                                        bn_ch = ch_out_
                                    else:
                                        bn_ch = ch_in_
                                    self.bn[ii][jj][kk] = BatchNorm(
                                        bn_ch, trainable=self.trainable)
                                    self.log.info('Init SD: {}'.format(
                                        self.compute_std(
                                            [f_, f_, ch_in_, ch_out_])))
                                self.log.info('Filter: {}'.format(
                                    [f_, f_, ch_in_, ch_out_]))
                                self.log.info('Weights: {}'.format(
                                    self.w[ii][jj][kk].name))
                                self.log.info('Trainable: {}'.format(
                                    self.trainable))
                                pass
                            pass
                        pass
                    pass
                pass
            pass
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['input']
        phase_train = inp['phase_train']
        prev_inp = x
        with tf.variable_scope(self.scope):
            for ii in xrange(self.num_stage):
                self.log.info(
                    'Stage count {:d} of {:d}'.format(ii, self.num_stage))
                ch_in = self.channels[ii]
                ch_out = self.channels[ii + 1]
                s = self.strides[ii]
                with tf.variable_scope('stage_{}'.format(ii)):
                    for jj in xrange(self.layers[ii]):
                        self.log.info('Layer count {:d} of {:d}'.format(
                            jj, self.layers[ii]))
                        h = prev_inp
                        if jj > 0:
                            ch_in = ch_out
                        else:
                            # First unit of the layer.
                            self.log.info(
                                'In {:d} Out {:d}'.format(ch_in, ch_out))
                            if self.compatible:
                                # In compatible mode, always project.
                                with tf.variable_scope('shortcut'):
                                    prev_inp = self.apply_shortcut(
                                        prev_inp, ch_in, ch_out,
                                        phase_train=phase_train,
                                        w=self.shortcut_w[ii],
                                        bn=self.shortcut_bn[ii],
                                        stride=s)
                            else:
                                if ch_in != ch_out:
                                    with tf.variable_scope('shortcut'):
                                        prev_inp = self.apply_shortcut(
                                            prev_inp, ch_in, ch_out,
                                            phase_train=phase_train,
                                            w=self.shortcut_w[ii],
                                            bn=self.shortcut_bn[ii],
                                            stride=s)
                                elif s != 1:
                                    if not self.dilation:
                                        prev_inp = AvgPool(s)(prev_inp)
                                        self.log.info('After pool shape: {}'.format(
                                            prev_inp.get_shape()))

                        with tf.variable_scope('layer_{}'.format(jj)):

                            if self.compatible:
                                # A compatible graph for building weights
                                # older version of ResNet
                                if self.bottleneck:
                                    with tf.variable_scope('unit_0'):
                                        f_, ch_in_, ch_out_ = \
                                            self.compute_in_out(
                                                0, ch_in, ch_out)
                                        h = Conv2D(self.w[ii][jj][0],
                                                   stride=s)(h)
                                        h = self.bn[ii][jj][0](
                                            {'input': h,
                                             'phase_train': phase_train})
                                        h = tf.nn.relu(h)
                                    with tf.variable_scope('unit_1'):
                                        f_, ch_in_, ch_out_ = \
                                            self.compute_in_out(
                                                1, ch_in, ch_out)
                                        h = Conv2D(self.w[ii][jj][1])(h)
                                        h = self.bn[ii][jj][1](
                                            {'input': h,
                                             'phase_train': phase_train})
                                        h = tf.nn.relu(h)
                                    with tf.variable_scope('unit_2'):
                                        f_, ch_in_, ch_out_ = \
                                            self.compute_in_out(
                                                2, ch_in, ch_out)
                                        h = Conv2D(self.w[ii][jj][2])(h)
                                        h = self.bn[ii][jj][2](
                                            {'input': h,
                                             'phase_train': phase_train})
                                else:
                                    with tf.variable_scope('unit_0'):
                                        f_, ch_in_, ch_out_ = \
                                            self.compute_in_out(
                                                0, ch_in, ch_out)
                                        h = Conv2D(self.w[ii][jj][0],
                                                   stride=s)(h)
                                        h = self.bn[ii][jj][0](
                                            {'input': h,
                                             'phase_train': phase_train})
                                        h = tf.nn.relu(h)
                                    with tf.variable_scope('unit_1'):
                                        f_, ch_in_, ch_out_ = \
                                            self.compute_in_out(
                                                1, ch_in, ch_out)
                                        h = Conv2D(self.w[ii][jj][1])(h)
                                        h = self.bn[ii][jj][1](
                                            {'input': h,
                                             'phase_train': phase_train})
                                s = 1
                            else:
                                # New version of ResNet
                                # Full pre-activation
                                for kk in xrange(self.unit_depth):
                                    with tf.variable_scope('unit_{}'.format(kk)):
                                        f_, ch_in_, ch_out_ = self.compute_in_out(
                                            kk, ch_in, ch_out)
                                        h = self.bn[ii][jj][kk](
                                            {'input': h,
                                             'phase_train': phase_train})
                                        h = tf.nn.relu(h)
                                        if self.dilation:
                                            h = DilatedConv2D(
                                                self.w[ii][jj][kk], rate=s)(h)
                                        else:
                                            h = Conv2D(self.w[ii][jj][kk],
                                                       stride=s)(h)
                                        self.log.info('Unit {} shape: {}'.format(
                                            kk, h.get_shape()))
                                        pass
                                    if not self.dilation and kk == 0:
                                        s = 1
                            pass

                        if self.compatible:
                            # Old version
                            # Relu after add
                            prev_inp = tf.nn.relu(prev_inp + h)
                            if not self.has_var('stage_{}/layer_{}/relu'.format(ii, jj)):
                                self.register_var(
                                    'stage_{}/layer_{}/relu'.format(ii, jj), prev_inp)
                                # print 'stage_{}/layer_{}/relu'.format(ii, jj)
                        else:
                            # New version
                            # Pure linear
                            prev_inp = prev_inp + h
                        self.log.info('After add shape: {}'.format(
                            prev_inp.get_shape()))
                        pass
                    pass
                pass
            pass
        return prev_inp

    def get_save_var_dict(self):
        results = {}
        for ii in xrange(self.num_stage):
            stage_prefix = 'stage_{}/'.format(ii)
            if self.shortcut_w[ii] is not None:
                short_prefix = stage_prefix + 'shortcut/'
                results[stage_prefix + 'shortcut/w'] = self.shortcut_w[ii]
            if self.shortcut_bn[ii] is not None:
                self.add_prefix_to(
                    short_prefix + 'bn',
                    self.shortcut_bn[ii].get_save_var_dict(),
                    results)
            for jj in xrange(self.layers[ii]):
                for kk in xrange(self.unit_depth):
                    prefix = 'stage_{}/layer_{}/unit_{}/'.format(ii, jj, kk)
                    results[prefix + 'w'] = self.w[ii][jj][kk]
                    bn = self.bn[ii][jj][kk]
                    if bn is not None:
                        self.add_prefix_to(
                            prefix + 'bn', bn.get_save_var_dict(), results)
                    pass
                pass
            pass
        return results
    pass

if __name__ == '__main__':
    # 34-layer ResNet
    x = tf.placeholder('float', [3, 224, 224, 3], name='x')
    phase_train = tf.placeholder('bool', name='phase_train')
    w1 = tf.Variable(tf.truncated_normal_initializer(
        stddev=0.01)([7, 7, 3, 64]), name='w1')
    print w1
    b1 = tf.Variable(tf.truncated_normal_initializer(
        stddev=0.01)([64]), name='b1')
    h1 = Conv2D(w1, stride=2)(x) + b1
    bn1 = BatchNorm(64)
    h1 = bn1({'input': h1, 'phase_train': phase_train})
    h1 = MaxPool(2)(h1)

    hn = ResNet(layers=[3, 4, 6, 3],
                bottleneck=True,
                shortcut='projection',
                channels=[64, 64, 128, 256, 512],
                strides=[1, 2, 2, 2])(
        {'input': h1, 'phase_train': phase_train})
    print hn.get_shape()
    hn = AvgPool(7)(hn)
    print hn.get_shape()

    hn = ResNet(layers=[3, 4, 6, 3],
                bottleneck=False,
                shortcut='projection',
                channels=[64, 64, 128, 256, 512],
                strides=[1, 2, 2, 2])(
        {'input': h1, 'phase_train': phase_train})
    print hn.get_shape()
    hn = AvgPool(7)(hn)
    print hn.get_shape()

    hn = ResNet(layers=[3, 4, 6, 3],
                bottleneck=False,
                dilation=True,
                shortcut='projection',
                channels=[64, 64, 128, 256, 512],
                strides=[1, 2, 4, 8])(
        {'input': h1, 'phase_train': phase_train})
    print hn.get_shape()

    hn = ResNet(layers=[3, 4, 6, 3],
                bottleneck=True,
                dilation=True,
                shortcut='projection',
                channels=[64, 64, 128, 256, 512],
                strides=[1, 2, 4, 8])(
        {'input': h1, 'phase_train': phase_train})
    print hn.get_shape()

    hn = ResNet(layers=[3, 4, 6, 3],
                bottleneck=True,
                dilation=False,
                shortcut='identity',
                channels=[64, 64, 128, 256, 512],
                strides=[1, 2, 2, 2])(
        {'input': h1, 'phase_train': phase_train})
    print hn.get_shape()
