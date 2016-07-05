from tfplus.utils import logger
import tensorflow as tf


class GraphBuilder(object):

    def __init__(self):
        self.log = logger.get()
        self._var_dict = {}
        self._has_init_var = False
        pass

    def __call__(self, inp):
        return self.build(inp)

    def init_var(self):
        pass

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
                    name=None, trainable=True):
        """Initialize weights.

        Args:
            shape: shape of the weights, list of int
            wd: weight decay
        """
        if initializer is None:
            initializer = tf.truncated_normal_initializer(stddev=0.01)
        if init_val is None:
            var = tf.Variable(
                initializer(shape), name=name, trainable=trainable)
        else:
            var = tf.Variable(init_val, name=name, trainable=trainable)
        if wd is not None:
            weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
            tf.add_to_collection('losses', weight_decay)
        return var

    def get_single_input(self, inp):
        if type(inp) == dict:
            return inp['x']
        else:
            return inp
    pass

if __name__ == '__main__':
    GraphBuilder()
