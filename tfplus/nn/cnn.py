from graph_builder import GraphBuilder

import tensorflow as tf

from ops import Conv2D, MaxPool
from batch_norm import BatchNorm


class CNN(GraphBuilder):

    def __init__(self, f, ch, pool, act, use_bn, wd=None,
                 scope='cnn', init_weights=None, frozen=None,
                 shared_weights=None):
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
        self.init_weights = init_weights
        self.frozen = frozen
        self.shared_weights = shared_weights

        self.nlayers = len(f)
        self.w = [None] * self.nlayers
        self.b = [None] * self.nlayers
        self.batch_norm = [None] * self.nlayers
        self.num_copies = 0

        super(CNN, self).__init__()

        self.log.info('CNN: {}'.format(scope))
        self.log.info('Channels: {}'.format(ch))
        self.log.info('Activation: {}'.format(act))
        self.log.info('Pool: {}'.format(pool))
        self.log.info('BN: {}'.format(use_bn))
        self.log.info('Shared weights: {}'.format(shared_weights))
        pass

    def init_var(self):
        """Initialize variables."""
        f = self.filter_size
        ch = self.channels
        wd = self.wd
        with tf.variable_scope(self.scope):
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    if self.init_weights:
                        init = tf.constant_initializer
                    else:
                        init = None

                    if self.init_weights is not None and \
                            self.init_weights[ii] is not None:
                        init_val_w = init_weights[ii]['w']
                        init_val_b = init_weights[ii]['b']
                    else:
                        init_val_w = None
                        init_val_b = None

                    if self.frozen is not None and self.frozen[ii]:
                        trainable = False
                    else:
                        trainable = True

                    if self.shared_weights:
                        self.w[ii] = self.shared_weights[ii]['w']
                        self.b[ii] = self.shared_weights[ii]['b']
                    else:
                        self.w[ii] = self.declare_var(
                            [f[ii], f[ii], ch[ii], ch[ii + 1]],
                            name='w',
                            init_val=init_val_w, wd=wd,
                            trainable=trainable)
                        self.b[ii] = self.declare_var(
                            [ch[ii + 1]], init_val=init_val_b,
                            name='b',
                            trainable=trainable)

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
        with tf.variable_scope(self.scope):
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    out_ch = self.channels[ii + 1]

                    if ii == 0:
                        prev_inp = x
                    else:
                        prev_inp = h[ii - 1]

                    # h[ii] = conv2d(prev_inp, self.w[ii]) + self.b[ii]
                    h[ii] = Conv2D(self.w[ii])(prev_inp) + self.b[ii]

                    if self.use_bn[ii]:
                        if self.frozen is not None and self.frozen[ii]:
                            self.bn_frozen = True
                        else:
                            self.bn_frozen = False

                        if self.init_weights is not None and \
                                self.init_weights[ii] is not None:
                            init_beta = self.init_weights[ii][
                                'beta_{}'.format(self.num_copies)]
                            init_gamma = self.init_weights[ii][
                                'gamma_{}'.format(self.num_copies)]
                        else:
                            init_beta = None
                            init_gamma = None

                        self.batch_norm[ii] = BatchNorm(out_ch,
                                                        init_beta=init_beta,
                                                        init_gamma=init_gamma)
                        h[ii] = self.batch_norm[ii](
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
    pass
