from graph_builder import GraphBuilder

import tensorflow as tf

from ops import Conv2D, MaxPool
from batch_norm import BatchNorm


class DCNN(GraphBuilder):

    def __init__(self, f, ch, pool, act, use_bn, skip_ch=None, wd=None, scope='dcnn', trainable=True):
        self.filter_size = f
        self.channels = ch
        self.pool = pool
        self.act = act
        self.use_bn = use_bn
        self.wd = wd
        self.scope = scope
        self.trainable = trainable
        self.skip_ch = skip_ch

        self.nlayers = len(f)
        self.w = [None] * self.nlayers
        self.b = [None] * self.nlayers
        self.batch_norm = []
        self.num_copies = 0

        super(DCNN, self).__init__()

        self.log.info('DCNN: {}'.format(scope))
        self.log.info('Channels: {}'.format(ch))
        self.log.info('Activation: {}'.format(act))
        self.log.info('Unpool: {}'.format(pool))
        self.log.info('Skip channels: {}'.format(skip_ch))
        self.log.info('BN: {}'.format(use_bn))
        pass

    def init_var(self):
        f = self.filter_size
        ch = self.channels
        wd = self.wd
        self.batch_norm.append([None] * self.nlayers)
        trainable = self.trainable
        with tf.variable_scope(self.scope):
            in_ch = ch[0]
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    out_ch = ch[ii + 1]
                    if self.skip_ch is not None:
                        if self.skip_ch[ii] is not None:
                            in_ch += self.skip_ch[ii]
                    self.w[ii] = self.declare_var([f[ii], f[ii], out_ch, in_ch],
                                                  name='w', wd=wd,
                                                  trainable=trainable)
                    self.b[ii] = self.declare_var([out_ch],  name='b',
                                                  trainable=trainable)
                    self.log.info('Filter: {}, Trainable: {}'.format(
                        [f[ii], f[ii], out_ch, in_ch], trainable))
                    in_ch = out_ch
                    pass
                pass
            pass
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['input']
        phase_train = inp['phase_train']
        skip = inp['skip']

        with tf.variable_scope(self.scope):
            h = [None] * self.nlayers
            out_shape = [None] * self.nlayers
            batch = tf.shape(x)[0: 1]
            inp_size = tf.shape(x)[1: 3]
            cum_pool = 1

            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    cum_pool *= self.pool[ii]
                    out_ch = self.channels[ii + 1]

                    if ii == 0:
                        prev_inp = x
                    else:
                        prev_inp = h[ii - 1]

                    if skip is not None:
                        if skip[ii] is not None:
                            if ii == 0:
                                prev_inp = tf.concat(3, [prev_inp, skip[ii]])
                            else:
                                prev_inp = tf.concat(3, [prev_inp, skip[ii]])

                    out_shape[ii] = tf.concat(
                        0, [batch, inp_size * cum_pool, tf.constant([out_ch])])

                    h[ii] = tf.nn.conv2d_transpose(
                        prev_inp, self.w[ii], out_shape[ii],
                        strides=[1, self.pool[ii], self.pool[ii], 1]) + self.b[ii]

                    if self.use_bn[ii]:
                        self.batch_norm[self.num_copies][
                            ii] = BatchNorm(out_ch)
                        h[ii] = self.batch_norm[self.num_copies][ii](
                            {'input': h[ii], 'phase_train': phase_train})

                    if self.act[ii] is not None:
                        h[ii] = self.act[ii](h[ii])

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
