from graph_builder import GraphBuilder
from batch_norm import BatchNorm
from ops import Conv2D, DilatedConv2D, MaxPool, AvgPool

import tensorflow as tf


class ResNet(GraphBuilder):
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
                 shortcut='projection'):
        super(ResNet, self).__init__()
        self.channels = channels
        self.layers = layers
        self.strides = strides
        self.scope = scope
        self.num_stage = len(layers)
        self.w = [None] * self.num_stage
        self.proj_w = [None] * self.num_stage
        self.b = [None] * self.num_stage
        self.bottleneck = bottleneck
        self.dilation = dilation
        self.shortcut = shortcut
        self.wd = wd
        if bottleneck:
            self.unit_depth = 3
        else:
            self.unit_depth = 2
        pass

    def compute_in_out(self, kk, ch_in, ch_out):
        if self.bottleneck:
            if kk == 0:
                f_ = 1
                ch_in_ = ch_in
                ch_out_ = ch_out / 4
            elif kk == 1:
                f_ = 3
                ch_in_ = ch_out / 4
                ch_out_ = ch_out / 4
            else:
                f_ = 1
                ch_in_ = ch_out / 4
                ch_out_ = ch_out
        else:
            f_ = 3
            if kk == 0:
                ch_in_ = ch_in
            else:
                ch_in_ = ch_out
            ch_out_ = ch_out
        return f_, ch_in_, ch_out_

    def apply_shortcut(self, prev_inp, ch_in, ch_out, proj_w=None, stride=None):
        if self.shortcut == 'projection':
            if self.dilation:
                prev_inp = DilatedConv2D(proj_w, rate=stride)(prev_inp)
            else:
                prev_inp = Conv2D(proj_w, stride=stride)(prev_inp)
        elif self.shortcut == 'identity':
            pad_ch = ch_out - ch_in
            if pad_ch < 0:
                raise Exception('Must use projection when ch_in > ch_out.')
            prev_inp = tf.pad(prev_inp, [[0, 0], [0, 0], [0, 0], [0, pad_ch]])
            if stride > 1:
                prev_inp = AvgPool(stride)(prev_inp)
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
                    self.b[ii] = [None] * self.layers[ii]
                    for jj in xrange(self.layers[ii]):
                        if jj > 0:
                            ch_in = ch_out
                            pass
                        self.w[ii][jj] = [None] * self.unit_depth
                        self.b[ii][jj] = [None] * self.unit_depth
                        if ch_in != ch_out:
                            if self.shortcut == 'projection':
                                self.proj_w[ii] = self.declare_var(
                                    [1, 1, ch_in, ch_out], wd=self.wd,
                                    name='proj_w')
                                pass
                            pass
                        with tf.variable_scope('layer_{}'.format(jj)):
                            for kk in xrange(self.unit_depth):
                                with tf.variable_scope('unit_{}'.format(kk)):
                                    f_, ch_in_, ch_out_ = self.compute_in_out(
                                        kk, ch_in, ch_out)
                                    self.w[ii][jj][kk] = self.declare_var(
                                        [f_, f_, ch_in_, ch_out_], wd=self.wd,
                                        name='w')
                                    self.b[ii][jj][kk] = self.declare_var(
                                        [ch_out_], wd=self.wd, name='b')
                                self.log.info('Filter: {}'.format(
                                    [f_, f_, ch_in_, ch_out_]))
                                self.log.info('Weights: {}'.format(
                                    self.w[ii][jj][kk].name))
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
        self.bn = [None] * self.num_stage
        with tf.variable_scope(self.scope):
            for ii in xrange(self.num_stage):
                print 'Stage count', ii, 'of', self.num_stage
                ch_in = self.channels[ii]
                ch_out = self.channels[ii + 1]
                s = self.strides[ii]
                with tf.variable_scope('stage_{}'.format(ii)):
                    self.bn[ii] = [None] * self.layers[ii]
                    for jj in xrange(self.layers[ii]):
                        print 'Layer count', jj, 'of', self.layers[ii]
                        h = prev_inp
                        if jj > 0:
                            ch_in = ch_out
                        else:
                            # First unit of the layer.
                            if ch_in != ch_out:
                                prev_inp = self.apply_shortcut(
                                    prev_inp, ch_in, ch_out,
                                    proj_w=self.proj_w[ii],
                                    stride=s)
                            elif s != 1:
                                if not self.dilation:
                                    prev_inp = AvgPool(s)(prev_inp)
                                    self.log.info('After pool shape: {}'.format(
                                        prev_inp.get_shape()))

                        with tf.variable_scope('layer_{}'.format(jj)):
                            self.bn[ii][jj] = [None] * self.unit_depth
                            for kk in xrange(self.unit_depth):
                                with tf.variable_scope('unit_{}'.format(kk)):
                                    f_, ch_in_, ch_out_ = self.compute_in_out(
                                        kk, ch_in, ch_out)
                                    self.bn[ii][jj][kk] = BatchNorm(ch_in_)
                                    h = self.bn[ii][jj][kk](
                                        {'input': h,
                                         'phase_train': phase_train})
                                    h = tf.nn.relu(h)
                                    if self.dilation:
                                        h = DilatedConv2D(
                                            self.w[ii][jj][kk], rate=s)(
                                            h) + self.b[ii][jj][kk]
                                    else:
                                        h = Conv2D(self.w[ii][jj][kk],
                                                   stride=s)(
                                            h) + self.b[ii][jj][kk]
                                    self.log.info('Unit {} shape: {}'.format(
                                        kk, h.get_shape()))
                                    pass
                                if not self.dilation and kk == 0:
                                    # Change the stride to 1 after 2.
                                    # Only in standard mode.
                                    # In dilation mode, everything is accumulated.
                                    # Nothing is down-sampled.
                                    # i.e. 1,1,1,2,2,2,4,4,4,8,8,8
                                    s = 1
                            pass
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
            if self.proj_w[ii] is not None:
                results[stage_prefix + 'proj_w'] = self.proj_w[ii]
            for jj in xrange(self.layers[ii]):
                for kk in xrange(self.unit_depth):
                    prefix = 'stage_{}/layer_{}/unit_{}/'.format(ii, jj, kk)
                    results[prefix + 'w'] = self.w[ii][jj][kk]
                    results[prefix + 'b'] = self.b[ii][jj][kk]
                    bn = self.bn[ii][jj][kk]
                    if bn is not None:
                        results[prefix + 'bn/beta'] = bn.beta
                        results[prefix + 'bn/gamma'] = bn.gamma
                        ema_mean, ema_var = bn.get_shadow_ema()
                        results[prefix + 'bn/ema_mean'] = ema_mean
                        results[prefix + 'bn/ema_var'] = ema_var
                        pass
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
