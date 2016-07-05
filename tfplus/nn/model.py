from graph_builder import GraphBuilder

import os
import tensorflow as tf

from tfplus.utils import assign_model_id, OptionBase, Saver, Factory
from tfplus.utils import logger

_factory = None


def get_factory():
    global _factory
    if _factory is None:
        _factory = Factory()
        pass
    return _factory


def register(name, cls):
    return get_factory().register(name, cls)


def create(_clsname, **kwargs):
    return get_factory().create(_clsname, **kwargs)


def create_from_main(_clsname, **kwargs):
    return get_factory().create_from_main(_clsname, **kwargs)


def gen_id(name):
    return '{}-{}'.format(name, assign_model_id.get_id())

_model_name_registry = {}


class Model(GraphBuilder, OptionBase):
    """Tensorflow model abstract object."""

    def __init__(self, name='model'):
        GraphBuilder.__init__(self)
        OptionBase.__init__(self)
        self._inp_var_dict = {}
        self._var_dict = {}
        self._loss = None
        self._gpu = -1
        counter = 1
        _name = name
        while _name in _model_name_registry:
            _name = name + '_{:d}'.format(counter)
        self._name = _name
        self.log.info('Registering model name "{}"'.format(_name))
        self._ckpt_fname = None
        self._saver = None
        pass

    @property
    def gpu(self):
        return self._gpu

    def set_gpu(self, value):
        self._gpu = value
        return self

    @property
    def name(self):
        return self._name

    @property
    def folder(self):
        return self._folder

    def set_folder(self, value):
        self._folder = value

    def get_folder(self):
        return self._folder

    @property
    def device(self):
        if self.gpu > 0:
            return '/gpu:{}'.format(self.gpu)
        else:
            return '/cpu:0'

    def get_name(self):
        return self._name

    def set_name(self, value):
        self._name = value
        return self

    def get_cpu_list(self):
        return ['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin',
                'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense',
                'BatchMatMul', 'Gather', 'Print']

    def get_device_fn(self):
        """Choose device for different ops."""
        OPS_ON_CPU = set(self.get_cpu_list())

        def _device_fn(op):
            if op.type in OPS_ON_CPU:
                return "/cpu:0"
            else:
                # Other ops will be placed on GPU if available, otherwise CPU.
                return self.device
        return _device_fn

    def restore_options_from(self, folder):
        """Restore all model options, not the weights.
        WARNING: Call only before built the graph.
        """
        if folder is not None:
            self.set_all_options(self.read_options(folder, self.name))
        return self

    def restore_weights_from(self, sess, folder):
        """Restore all model options, not the weights.
        WARNING: Call only after built the graph.
        """
        if folder is not None:
            Saver(folder, var_dict=self.get_save_var_dict(),
                  fname=self.name).restore(sess)
        return self

    @property
    def saver(self):
        if self._saver is None:
            self._saver = Saver(self.folder,
                                var_dict=self.get_save_var_dict(),
                                fname=self.name)
        return self._saver

    def save(self, sess, step=0):
        """Save model.

        Args:
            sess: tensorflow session
            step: number of steps
        """
        self.save_options(self.folder, self.name)
        self.saver.save(sess, global_step=step)
        pass

    def has_input_var(self, name):
        """Checks whether an input variable exists.

        Args:
            name: variable name.

        Returns:
            exists: bool.
        """
        return name in self._inp_var_dict

    def get_input_var(self, name):
        """Get an input variable.

        Args:
            name: variable name.

        Returns:
            var: variable.
        """
        return self._inp_var_dict[name]

    def get_all_input_vars(self):
        return self._inp_var_dict

    def register_input_var(self, name, var):
        """Register an input variable into collection.

        Args:
            name: variable
            var: variable object.
        """

        if name in self._inp_var_dict:
            raise Exception('Variable name already exists: {}'.format(name))
        self._inp_var_dict[name] = var
        self.register_var(name, var)
        pass

    def add_input_var(self, name, shape, dtype='float'):
        """Declare an input variable and register it.

        Args:
            name: variable name.
            shape: variable shape.
            dtype: variable dtype.

        Returns:
            var: variable object.
        """
        var = tf.placeholder(dtype, shape, name=name)
        self.register_input_var(name, var)
        return var

    def get_loss(self):
        """Get loss variable."""
        if self._loss is None:
            self._loss = tf.add_n(tf.get_collection('losses'), name='loss')
        else:
            self.log.fatal("Calling get_loss twice")
        return self._loss

    def add_loss(self, var):
        """Add a loss."""
        tf.add_to_collection('losses', var)

    def build_input(self):
        """Build input nodes. To be implemented by subclasses.

        Returns:
            input_var: dict.
        """
        raise Exception('Not implemented')

    def build(self, inp):
        """Build computation graph. To be implemented by subclasses.

        Args:
            inp: dict or single variable.

        Returns:
            output_var: dict or single variable.
        """
        raise Exception('Not implemented')

    def build_loss_grad(self, inp, output):
        """Build gradient graph. To be implemented by subclasses.

        """
        raise Exception('Not implemented')

    def build_eval(self):
        """Build nodes for evaluation/inference."""
        inp_var = self.build_input()
        self.build(inp_var)
        return self

    def build_all(self):
        """Build all nodes."""
        with tf.device(self.get_device_fn()):
            inp_var = self.build_input()
            output_var = self.build(inp_var)
            self.build_loss_grad(inp_var, output_var)
        return self

    def add_prefix_to(self, prefix, from_, to):
        for key in from_.iterkeys():
            to[prefix + '/' + key] = from_[key]

    def get_save_var_dict(self):
        """Get a dictionary of variables to restore."""
        raise Exception('Not implemented')
    pass


class ContainerModel(Model):

    def __init__(self, name='model'):
        super(ContainerModel, self).__init__(name=name)
        self.sub_models = []
        pass

    def add_sub_model(self, m):
        self.sub_models.add(m)
        pass

    def save(self):
        for m in self.sub_models:
            m.save()
            pass
        return super(ContainerModel, self).save()

    def save_options(self):
        for m in self.sub_models:
            m.save_options()
            pass
        return super(ContainerModel, self).save_options()

    def restore_options_from(self, folder):
        for m in self.sub_models:
            m.restore_options_from(folder)
            pass
        return super(ContainerModel, self).restore_from(folder)
