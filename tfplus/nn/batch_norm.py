from graph_builder import GraphBuilder
import tensorflow as tf


class BatchNorm(GraphBuilder):

    def __init__(self, n_out, scope='bn',
                 affine=True, init_beta=None, init_gamma=None, frozen=False):
        super(BatchNorm, self).__init__()
        self.n_out = n_out
        self.scope = scope
        self.affine = affine
        self.init_beta = init_beta
        self.init_gamma = init_gamma
        self.frozen = frozen
        pass

    def init_var(self):
        trainable = not self.frozen
        with tf.variable_scope(self.scope):
            if self.init_beta is None:
                self.init_beta = tf.constant(0.0, shape=[self.n_out])
            if self.init_gamma is None:
                self.init_gamma = tf.constant(1.0, shape=[self.n_out])

            self.beta = self.declare_var(
                [self.n_out], init_val=self.init_beta, name='beta',
                trainable=trainable)
            self.gamma = self.declare_var(
                [self.n_out], init_val=self.init_gamma, name='gamma',
                trainable=trainable)
            pass
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['input']
        phase_train = inp['phase_train']

        with tf.variable_scope(self.scope):
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name='moments')
            batch_mean.set_shape([self.n_out])
            batch_var.set_shape([self.n_out])

            phase_train_f = tf.to_float(phase_train)
            decay = 1 - 0.1 * phase_train_f
            ema = tf.train.ExponentialMovingAverage(decay=decay)

            def mean_var_with_update():
                ema_apply_op_local = ema.apply([batch_mean, batch_var])
                with tf.control_dependencies([ema_apply_op_local]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            def mean_var_no_update():
                ema_mean_local, ema_var_local = ema.average(
                    batch_mean), ema.average(batch_var)
                return ema_mean_local, ema_var_local

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                mean_var_no_update)
            normed = tf.nn.batch_normalization(
                x, mean, var, self.beta, self.gamma, 1e-3)
        return normed
    pass
