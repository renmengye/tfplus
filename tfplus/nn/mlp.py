from graph_builder import GraphBuilder
from batch_norm import BatchNorm

import numpy as np
import tensorflow as tf


class MLP(GraphBuilder):

    def __init__(self, dims, act, use_bn=None, add_bias=True, wd=None,
                 scope='mlp', init_weights=None, init_stddev=0.01, frozen=None):
        """Add MLP. N = number of layers.

        Args:
            dims: layer-wise dimensions, list of N int
            act: activation function, list of N function
            wd: weight decay
        """
        self.nlayers = len(dims) - 1
        self.w = [None] * self.nlayers
        self.b = [None] * self.nlayers
        self.dims = dims
        self.act = act
        self.add_bias = add_bias
        self.wd = wd
        self.scope = scope
        self.init_weights = init_weights
        self.frozen = frozen
        self.init_stddev = init_stddev
        if use_bn is None:
            self.use_bn = [False] * self.nlayers
            self.bn = [None] * self.nlayers
        else:
            self.use_bn = use_bn
            self.bn = [None] * self.nlayers

        super(MLP, self).__init__()

        self.log.info('MLP: {}'.format(scope))
        self.log.info('Dimensions: {}'.format(dims))
        self.log.info('Activation: {}'.format(act))
        self.log.info('Add bias: {}'.format(add_bias))
        pass

    def init_var(self):
        self._has_init_var = True
        dims = self.dims
        wd = self.wd
        with tf.variable_scope(self.scope):
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    nin = dims[ii]
                    nout = dims[ii + 1]

                    if self.init_weights is not None and \
                            self.init_weights[ii] is not None:
                        init_val_w = self.init_weights[ii]['w']
                        init_val_b = self.init_weights[ii]['b']
                    else:
                        init_val_w = None
                        init_val_b = None

                    if self.frozen is not None and self.frozen[ii]:
                        trainable = False
                    else:
                        trainable = True

                    if self.use_bn[ii]:
                        self.bn[ii] = BatchNorm(nin, [0], decay=0.999)

                    if self.init_stddev is None:
                        self.log.info('Using Xavier initialization')
                        std = 1 / np.sqrt(nin)
                        self.log.info('Std: {:.4f}'.format(std))
                        self.log.info('Uniform')
                        dist = 'uniform'
                    else:
                        self.log.info('Using standard initialization')
                        std = self.init_stddev
                        self.log.info('Std: {:.4f}'.format(std))
                        self.log.info('Gaussian')
                        dist = 'normal'
                    self.w[ii] = self.declare_var(
                        [nin, nout],
                        init_val=init_val_w, wd=wd,
                        name='w',
                        trainable=trainable, stddev=std, dist=dist)
                    self.log.info('Weights: {} Trainable: {}'.format(
                        [nin, nout], trainable))
                    if self.add_bias:
                        self.b[ii] = self.declare_var(
                            [nout], init_val=init_val_b,
                            name='b',
                            trainable=trainable)
                        self.log.info('Bias: {} Trainable: {}'.format(
                            [nout], trainable))
                        pass
                    pass
                pass
            pass
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['input']
        phase_train = inp['phase_train']
        if 'dropout_keep' in inp:
            dropout_keep = inp['dropout_keep']
        else:
            dropout_keep = None
        h = [None] * self.nlayers
        with tf.variable_scope(self.scope):
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    if ii == 0:
                        prev_inp = x
                    else:
                        prev_inp = h[ii - 1]

                    if self.use_bn[ii]:
                        prev_inp = self.bn[ii](
                            {'input': prev_inp, 'phase_train': phase_train})

                    if dropout_keep is not None:
                        if dropout_keep[ii] is not None:
                            prev_inp = tf.nn.dropout(prev_inp,
                                                     dropout_keep[ii])

                    h[ii] = tf.matmul(prev_inp, self.w[ii])

                    if self.add_bias:
                        h[ii] += self.b[ii]
                        pass

                    if self.act[ii]:
                        h[ii] = self.act[ii](h[ii])
                        pass
                    pass
                pass
            pass
        return h[-1]

    def get_save_var_dict(self):
        results = {}
        for ii in xrange(self.nlayers):
            prefix = 'layer_{}/'.format(ii)
            results[prefix + 'w'] = self.w[ii]
            if self.add_bias:
                results[prefix + 'b'] = self.b[ii]
            if self.bn[ii] is not None:
                self.add_prefix_to(
                    prefix + 'bn', self.bn[ii].get_save_var_dict(), results)
            pass
        return results
    pass
