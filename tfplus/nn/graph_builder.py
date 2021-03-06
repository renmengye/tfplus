from tfplus.utils import logger
import tensorflow as tf


class GraphBuilder(object):

    def __init__(self):
        self.log = logger.get()
        self._var_dict = {}
        self._has_init_var = False
        self._variable_sharing = False
        pass

    def __call__(self, inp):
        return self.build(inp)

    def init_var(self):
        pass

    def set_variable_sharing(self, value):
        self._variable_sharing = value

    @property
    def variable_sharing(self):
        return self._variable_sharing

    def lazy_init_var(self):
        if not self._has_init_var:
            self.init_var()
            self._has_init_var = True
        pass

    def build(self, inp):
        """Build computation graph. To be implemented by subclasses."""
        pass

    def register_var(self, name, var):
        """Register a variable.

        Args:
            name: variable name.
        """
        if name in self._var_dict:
            raise Exception('Variable name already exists: {}'.format(name))
        self._var_dict[name] = var
        pass

    def has_var(self, name):
        return name in self._var_dict

    def get_var(self, name):
        """Get a variable.

        Args:
            name: variable name.
        """
        return self._var_dict[name]

    def declare_var(self, shape, initializer=None, init_val=None, wd=None,
                    name=None, trainable=True, stddev=0.01, dist='normal'):
        """Initialize weights.

        Args:
            shape: shape of the weights, list of int
            wd: weight decay
        """
        if initializer is None:
            if stddev > 0:
                if dist == 'normal':
                    self.log.info('Truncated normal initialization')
                    initializer = tf.truncated_normal_initializer(
                        stddev=stddev)
                elif dist == 'uniform':
                    self.log.info('Uniform initialization')
                    initializer = tf.random_uniform_initializer(
                        minval=-stddev, maxval=stddev)
                else:
                    raise Exception('Unknown distribution "{}"'.format(dist))
            else:
                initializer = tf.constant_initializer(0.0)
        if init_val is None:
            if self.variable_sharing:
                with tf.device('/gpu:0'):
                # with tf.device('/cpu:0'):
                    var = tf.get_variable(name, shape, initializer=initializer)
            else:
                var = tf.Variable(
                    initializer(shape), name=name, trainable=trainable,
                    dtype='float')
        else:
            if self.variable_sharing:
                with tf.device('/gpu:0'):
                # with tf.device('/cpu:0'):
                    var = tf.get_variable(
                        name, shape,
                        initializer=tf.constant_initializer(init_val))
            else:
                var = tf.Variable(init_val, name=name, trainable=trainable,
                                  dtype='float')
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def get_single_input(self, inp):
        if type(inp) == dict:
            return inp['x']
        else:
            return inp

    def add_prefix_to(self, prefix, from_, to):
        for key in from_.iterkeys():
            if prefix is not None:
                newkey = prefix + '/' + key
            else:
                newkey = key
            to[newkey] = from_[key]
            pass
        return to

    pass

if __name__ == '__main__':
    GraphBuilder()
