from graph_builder import GraphBuilder
import tensorflow as tf


class BatchNorm(GraphBuilder):

    # def __init__(self, n_out, axes=[1], scope='ln',
    #              affine=True, init_beta=None, init_gamma=None, trainable=True,
    #              decay=0.9, eps=1e-3):
    #     super(BatchNorm, self).__init__()
    #     self.n_out = n_out
    #     self.scope = scope
    #     self.affine = affine
    #     self.init_beta = init_beta
    #     self.init_gamma = init_gamma
    #     self.trainable = trainable
    #     self.decay = decay
    #     self.axes = axes
    #     self.eps = eps
    #     pass

    # def init_var(self):
    #     trainable = self.trainable
    #     with tf.variable_scope(self.scope):
    #         if self.init_beta is None:
    #             self.init_beta = tf.constant(0.0, shape=[self.n_out])
    #         if self.init_gamma is None:
    #             self.init_gamma = tf.constant(1.0, shape=[self.n_out])
    #         self.beta = self.declare_var(
    #             [self.n_out], init_val=self.init_beta, name='beta',
    #             trainable=trainable)
    #         self.gamma = self.declare_var(
    #             [self.n_out], init_val=self.init_gamma, name='gamma',
    #             trainable=trainable)
    #         self.ema = tf.train.ExponentialMovingAverage(decay=self.decay)
    #         self.batch_mean = None
    #         self.batch_var = None
    #         self.ema_apply_op = None
    #         pass
    #     pass

    # def build(self, inp):
    #     self.lazy_init_var()
    #     x = inp['input']
    #     phase_train = inp['phase_train']

    #     with tf.variable_scope(self.scope):
    #         # Here only decalre batch_mean once.
    #         # If you want different statistics, construct two BatchNorm object.
    #         # Otherwise it will only take statistics from the first pass.
    #         if self.batch_mean is None:
    #             self.batch_mean, self.batch_var = tf.nn.moments(
    #                 x, self.axes, name='moments')
    #             self.batch_mean.set_shape([self.n_out])
    #             self.batch_var.set_shape([self.n_out])

    #         def mean_var_with_update():
    #             if self.ema_apply_op is None:
    #                 self.ema_apply_op = self.ema.apply(
    #                     [self.batch_mean, self.batch_var])
    #             with tf.control_dependencies([self.ema_apply_op]):
    #                 return tf.identity(self.batch_mean), \
    #                     tf.identity(self.batch_var)

    #         def mean_var_no_update():
    #             return self.get_shadow_ema()

    #         mean, var = tf.cond(phase_train,
    #                             mean_var_with_update,
    #                             mean_var_no_update)
    #         normed = tf.nn.batch_normalization(
    #             x, mean, var, self.beta, self.gamma, self.eps)
    #     return normed

    # def get_shadow_ema(self):
    #     return self.ema.average(self.batch_mean), \
    #         self.ema.average(self.batch_var)

    # def get_save_var_dict(self):
    #     ema_mean, ema_var = self.get_shadow_ema()
    #     return {'beta': self.beta, 'gamma': self.gamma, 'ema_mean': ema_mean,
    #             'ema_var': ema_var}
    pass
