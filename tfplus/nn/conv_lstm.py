from graph_builder import GraphBuilder
from ops import Conv2D

import tensorflow as tf


class ConvLSTM(GraphBuilder):

    def __init__(self, filter_size, inp_depth, hid_depth, wd=None, scope='lstm',
                 init_weights=None, frozen=False):
        super(ConvLSTM, self).__init__()
        self.filter_size = filter_size
        self.inp_depth = inp_depth
        self.hid_depth = hid_depth
        self.wd = wd
        self.scope = scope
        self.init_w = init_weights
        self.frozen = frozen
        pass

    def init_var(self):
        self.log.info('LSTM: {}'.format(self.scope))
        self.log.info('Input dim: {}'.format(self.inp_depth))
        self.log.info('Hidden dim: {}'.format(self.hid_depth))

        if self.init_w is None:
            self.init_w = {}
            for w in ['w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu',
                      'w_hu', 'b_u', 'w_xo', 'w_ho', 'b_o']:
                self.init_w[w] = None

        trainable = not self.frozen
        self.log.info('Trainable: {}'.format(trainable))

        with tf.variable_scope(self.scope):
            # Input gate
            self.w_xi = self.declare_var(
                [self.filter_size, self.filter_size, self.inp_depth,
                    self.hid_depth], init_val=self.init_w['w_xi'],
                wd=self.wd, name='w_xi', trainable=trainable)
            self.w_hi = self.declare_var(
                [self.filter_size, self.filter_size, self.hid_depth,
                    self.hid_depth], init_val=self.init_w['w_hi'],
                wd=self.wd, name='w_hi', trainable=trainable)
            self.b_i = self.declare_var(
                [self.hid_depth], init_val=self.init_w['b_i'],
                initializer=tf.constant_initializer(0.0),
                name='b_i', trainable=trainable)

            # Forget gate
            self.w_xf = self.declare_var(
                [self.filter_size, self.filter_size, self.inp_depth,
                    self.hid_depth], init_val=self.init_w['w_xf'],
                wd=self.wd, name='w_xf', trainable=trainable)
            self.w_hf = self.declare_var(
                [self.filter_size, self.filter_size, self.hid_depth,
                    self.hid_depth], init_val=self.init_w['w_hf'],
                wd=self.wd, name='w_hf', trainable=trainable)
            self.b_f = self.declare_var(
                [self.hid_depth], init_val=self.init_w['b_f'],
                initializer=tf.constant_initializer(1.0),
                name='b_f', trainable=trainable)

            # Input activation
            self.w_xu = self.declare_var(
                [self.filter_size, self.filter_size, self.inp_depth,
                    self.hid_depth], init_val=self.init_w['w_xu'],
                wd=self.wd, name='w_xu', trainable=trainable)
            self.w_hu = self.declare_var(
                [self.filter_size, self.filter_size, self.hid_depth,
                    self.hid_depth], init_val=self.init_w['w_hu'],
                wd=self.wd, name='w_hu', trainable=trainable)
            self.b_u = self.declare_var(
                [self.hid_depth], init_val=self.init_w['b_u'],
                initializer=tf.constant_initializer(0.0),
                name='b_u', trainable=trainable)

            # Output gate
            self.w_xo = self.declare_var(
                [self.filter_size, self.filter_size, self.inp_depth,
                    self.hid_depth], init_val=self.init_w['w_xo'],
                wd=self.wd, name='w_xo', trainable=trainable)
            self.w_ho = self.declare_var(
                [self.filter_size, self.filter_size,
                    self.hid_depth, self.hid_depth],
                init_val=self.init_w['w_ho'],
                wd=self.wd, name='w_ho', trainable=trainable)
            self.b_o = self.declare_var(
                [self.hid_depth], init_val=self.init_w['b_o'],
                initializer=tf.constant_initializer(0.0),
                name='b_o', trainable=trainable)
            pass

    def build(self, inp):
        """Build LSTM graph.

        Args:
            inp: input, state.

        Returns:
            results: state.
        """
        self.lazy_init_var()
        x = inp['input']
        state = inp['state']

        with tf.variable_scope(self.scope):
            c = tf.slice(state, [0, 0, 0, 0], [-1, -1, -1, self.hid_depth])
            h = tf.slice(state, [0, 0, 0, self.hid_depth],
                         [-1, -1, -1, self.hid_depth])
            g_i = tf.sigmoid(Conv2D(self.w_xi)(x) +
                             Conv2D(self.w_hi)(h) + self.b_i)
            g_f = tf.sigmoid(Conv2D(self.w_xf)(x) +
                             Conv2D(self.w_hf)(h) + self.b_f)
            g_o = tf.sigmoid(Conv2D(self.w_xo)(x) +
                             Conv2D(self.w_ho)(h) + self.b_o)
            u = tf.tanh(Conv2D(self.w_xu)(x) +
                        Conv2D(self.w_hu)(h) + self.b_u)
            c = g_f * c + g_i * u
            h = g_o * tf.tanh(c)
            state = tf.concat(3, [c, h])

        return {'state': state}

    def get_save_var_dict(self):
        results = {}
        for name, w in zip(
            ['w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu',
             'w_hu', 'b_u', 'w_xo', 'w_ho', 'b_o'],
            [self.w_xi, self.w_hi, self.b_i, self.w_xf, self.w_hf,
             self.b_f, self.w_xu, self.w_hu, self.b_u, self.w_xo, self.w_ho,
             self.b_o]):
            results[name] = w
            pass
        return results
    pass
