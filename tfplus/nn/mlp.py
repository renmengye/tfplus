from graph_builder import GraphBuilder

import tensorflow as tf

class MLP(GraphBuilder):

    def __init__(self, dims, act, add_bias=True, dropout_keep=None,
                 wd=None, scope='mlp', init_weights=None,
                 frozen=None):
        """Add MLP. N = number of layers.

        Args:
            dims: layer-wise dimensions, list of N int
            act: activation function, list of N function
            dropout_keep: keep prob of dropout, list of N float
            wd: weight decay
        """
        self.nlayers = len(dims) - 1
        self.w = [None] * self.nlayers
        self.b = [None] * self.nlayers
        self.dims = dims
        self.act = act
        self.add_bias = add_bias
        self.dropout_keep = dropout_keep
        self.wd = wd
        self.scope = scope
        self.init_weights = init_weights
        self.frozen = frozen

        super(MLP, self).__init__()
        
        self.log.info('MLP: {}'.format(scope))
        self.log.info('Dimensions: {}'.format(dims))
        self.log.info('Activation: {}'.format(act))
        self.log.info('Dropout: {}'.format(dropout_keep))
        self.log.info('Add bias: {}'.format(add_bias))
        pass

    def init_var(self):
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

                    self.w[ii] = self.declare_var(
                        [nin, nout],
                        init_val=init_val_w, wd=wd,
                        name='w',
                        trainable=trainable)
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
        h = [None] * self.nlayers
        with tf.variable_scope(self.scope):
            for ii in xrange(self.nlayers):
                with tf.variable_scope('layer_{}'.format(ii)):
                    if ii == 0:
                        prev_inp = x
                    else:
                        prev_inp = h[ii - 1]

                    if self.dropout_keep is not None:
                        if self.dropout_keep[ii] is not None:
                            prev_inp = nn.Dropout(self.dropout_keep[ii], 
                                phase_train)(prev_inp)

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
    pass
