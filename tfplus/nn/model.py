from graph_builder import GraphBuilder

import os
import tensorflow as tf

from tfplus.utils import cmd_args, assign_model_id, OptionBase, Saver, Factory
from tfplus.utils import logger

cmd_args.add('gpu', 'int', -1)
cmd_args.add('model_id', 'str', None)
cmd_args.add('restore', 'str', None)
cmd_args.add('results', 'str', '../results')

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


class Model(GraphBuilder, OptionBase):
    """Tensorflow model abstract object."""

    def __init__(self):
        GraphBuilder.__init__(self)
        OptionBase.__init__(self)
        self._inp_var_dict = {}
        self._var_dict = {}
        self._loss = None
        self._name = 'default'
        self._id = self.gen_model_id()
        self._restore = None
        self._saver = None
        self._results_folder = None
        self._restore_step = 0
        self.register_option('gpu')
        self.register_option('model_id')
        self.register_option('restore')
        self.register_option('results')
        pass

    def gen_model_id(self):
        return '{}-{}'.format(self.name, assign_model_id.get_id())

    @property
    def id(self):
        return self._id

    @property
    def name(self):
        return self._name

    @property
    def folder(self):
        return os.path.join(self._results_folder, self._id)

    def get_name(self):
        return self._name

    def set_name(self, value):
        self._name = value
        self._id = self.gen_model_id()
        return self

    def get_description(self):
        return 'Model'

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
                return self._device
        return _device_fn

    def _restore_from(self, folder):
        """Sets all basic info of restoration."""
        self._saver = Saver(folder)
        info = self._saver.get_ckpt_info()
        self._id = info['model_id']
        self._restore = folder
        self._ckpt_fname = info['ckpt_fname']
        self._restore_step = info['step']
        pass

    def restore_from(self, folder):
        """Restore a model for non-commandline usage.

        Args:
            folder: folder where the model is stored.
        """
        self._restore_from(folder)
        self.set_all_options(self.read_options(self.folder, 'model'))
        pass

    @property
    def restore_step(self):
        return self._restore_step

    @property
    def results_folder(self):
        return self._results_folder

    def parse_opt(self):
        """Parse the options from command line. Handles restoring logic."""
        opt = OptionBase.parse_opt(self)

        if opt['gpu'] > 0:
            self._device = '/gpu:{}'.format(opt['gpu'])
        else:
            self._device = '/cpu:0'

        # Output folder.
        if opt['results'] is not None:
            self._results_folder = opt['results']
            pass

        # Use previously stored model options.
        if opt['restore'] is not None:
            self._restore_from(opt['restore'])
            opt = self.read_options(self.folder, 'model')
            pass
        elif opt['model_id'] is not None:
            self._id = opt['model_id']
            pass

        # Remove experiment-ralted options.
        for name in ['gpu', 'model_id', 'restore', 'results']:
            if name in opt:
                del opt[name]
                pass
            pass
        self.log.info('Model options: {}'.format(opt))
        return opt

    def init(self, sess):
        """Initialize model.

        Args:
            sess: tensorflow session
        """
        if self._restore is not None:
            self._saver.restore(sess, self._ckpt_fname)
            pass
        else:
            sess.run(tf.initialize_all_variables())
            pass
        return self

    def save(self, sess, step=0):
        """Save model.

        Args:
            sess: tensorflow session
            step: number of steps
        """
        if self._saver is None:
            self._saver = Saver(self.folder)
            self.save_options(self.folder, 'model')
        self._saver.save(sess, global_step=step)
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
        inp_var = self.build_input()
        output_var = self.build(inp_var)
        self.build_loss_grad(inp_var, output_var)
        return self
    pass
