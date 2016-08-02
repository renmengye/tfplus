from graph_builder import GraphBuilder
import tensorflow as tf


class LSTM(GraphBuilder):

    def __init__(self, inp_dim, hid_dim, wd=None, scope='lstm',
                 init_weights=None, trainable=True):
        super(LSTM, self).__init__()
        self.inp_dim = inp_dim
        self.hid_dim = hid_dim
        self.wd = wd
        self.scope = scope
        self.init_w = init_weights
        self.trainable = trainable
        pass

    def init_var(self):
        self.log.info('LSTM: {}'.format(self.scope))
        self.log.info('Input dim: {}'.format(self.inp_dim))
        self.log.info('Hidden dim: {}'.format(self.hid_dim))

        if self.init_w is None:
            self.init_w = {}
            for w in ['w_xi', 'w_hi', 'b_i', 'w_xf', 'w_hf', 'b_f', 'w_xu',
                      'w_hu', 'b_u', 'w_xo', 'w_ho', 'b_o']:
                self.init_w[w] = None

        trainable = self.trainable
        self.log.info('Trainable: {}'.format(trainable))

        with tf.variable_scope(self.scope):
            # Input gate
            self.w_xi = self.declare_var(
                [self.inp_dim, self.hid_dim], init_val=self.init_w['w_xi'],
                wd=self.wd, name='w_xi', trainable=trainable)
            self.w_hi = self.declare_var(
                [self.hid_dim, self.hid_dim], init_val=self.init_w['w_hi'],
                wd=self.wd, name='w_hi', trainable=trainable)
            self.b_i = self.declare_var(
                [self.hid_dim], init_val=self.init_w['b_i'],
                initializer=tf.constant_initializer(0.0),
                name='b_i', trainable=trainable)

            # Forget gate
            self.w_xf = self.declare_var(
                [self.inp_dim, self.hid_dim], init_val=self.init_w['w_xf'],
                wd=self.wd, name='w_xf', trainable=trainable)
            self.w_hf = self.declare_var(
                [self.hid_dim, self.hid_dim], init_val=self.init_w['w_hf'],
                wd=self.wd, name='w_hf', trainable=trainable)
            self.b_f = self.declare_var(
                [self.hid_dim], init_val=self.init_w['b_f'],
                initializer=tf.constant_initializer(1.0),
                name='b_f', trainable=trainable)

            # Input activation
            self.w_xu = self.declare_var(
                [self.inp_dim, self.hid_dim], init_val=self.init_w['w_xu'],
                wd=self.wd, name='w_xu', trainable=trainable)
            self.w_hu = self.declare_var(
                [self.hid_dim, self.hid_dim], init_val=self.init_w['w_hu'],
                wd=self.wd, name='w_hu', trainable=trainable)
            self.b_u = self.declare_var(
                [self.hid_dim], init_val=self.init_w['b_u'],
                initializer=tf.constant_initializer(0.0),
                name='b_u', trainable=trainable)

            # Output gate
            self.w_xo = self.declare_var(
                [self.inp_dim, self.hid_dim], init_val=self.init_w['w_xo'],
                wd=self.wd, name='w_xo', trainable=trainable)
            self.w_ho = self.declare_var(
                [self.hid_dim, self.hid_dim],
                init_val=self.init_w['w_ho'],
                wd=self.wd, name='w_ho', trainable=trainable)
            self.b_o = self.declare_var(
                [self.hid_dim], init_val=self.init_w['b_o'],
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
            c = tf.slice(state, [0, 0], [-1, self.hid_dim])
            h = tf.slice(state, [0, self.hid_dim], [-1, self.hid_dim])
            g_i = tf.sigmoid(tf.matmul(x, self.w_xi) +
                             tf.matmul(h, self.w_hi) + self.b_i)
            g_f = tf.sigmoid(tf.matmul(x, self.w_xf) +
                             tf.matmul(h, self.w_hf) + self.b_f)
            g_o = tf.sigmoid(tf.matmul(x, self.w_xo) +
                             tf.matmul(h, self.w_ho) + self.b_o)
            u = tf.tanh(tf.matmul(x, self.w_xu) +
                        tf.matmul(h, self.w_hu) + self.b_u)
            c = g_f * c + g_i * u
            h = g_o * tf.tanh(c)
            state = tf.concat(1, [c, h])

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
