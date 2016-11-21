from graph_builder import GraphBuilder
import tensorflow as tf


class NewBatchNorm(GraphBuilder):

    def __init__(self, n_out, axes=[0, 1, 2], scope='nbn', decay=0.9,
                 trainable=True):
        super(NewBatchNorm, self).__init__()
        self.n_out = n_out
        self.scope = scope
        self.decay = decay
        self.axes = axes
        pass

    def init_var(self):
        with tf.variable_scope(self.scope):
            self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)
            self.batch_mean = None
            self.batch_var = None
            self.ema_apply_op = None
            pass
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['input']
        phase_train = inp['phase_train']

        with tf.variable_scope(self.scope):
            # Here only decalre batch_mean once.
            # If you want different statistics, construct two BatchNorm object.
            # Otherwise it will only take statistics from the first pass.
            if self.batch_mean is None:
                self.batch_mean, self.batch_var = tf.nn.moments(
                    x, self.axes, name='moments')
                self.batch_mean.set_shape([self.n_out])
                self.batch_var.set_shape([self.n_out])

            def mean_var_with_update():
                if self.ema_apply_op is None:
                    self.ema_apply_op = self.ema.apply(
                        [self.batch_mean, self.batch_var])
                with tf.control_dependencies([self.ema_apply_op]):
                    return tf.identity(self.batch_mean), \
                        tf.identity(self.batch_var)

            def mean_var_no_update():
                return self.get_shadow_ema()

            mean, var = tf.cond(phase_train,
                                mean_var_with_update,
                                mean_var_no_update)

            normed = tf.nn.batch_normalization(
                x, mean, var, tf.zeros([self.n_out]),
                tf.ones([self.n_out]), 1.0)
        return normed

    def get_shadow_ema(self):
        return self.ema.average(self.batch_mean), \
            self.ema.average(self.batch_var)

    def get_save_var_dict(self):
        ema_mean, ema_var = self.get_shadow_ema()
        return {
            'ema_mean': ema_mean,
            'ema_var': ema_var
        }
    pass
