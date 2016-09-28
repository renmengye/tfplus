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
        self._aux_saver = None
        self._has_init = False
        self._has_built_all = False
        self._folder = None
        self._global_step = None
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
        if not os.path.exists(value):
            os.makedirs(value)
        self._folder = value
        return self

    def get_folder(self):
        return self._folder

    @property
    def device(self):
        if self.gpu >= 0:
            return '/gpu:{}'.format(self.gpu)
        else:
            return '/cpu:0'

    def get_name(self):
        return self._name

    def set_name(self, value):
        if self.has_init:
            raise Exception('Cannot change name after initialization')
        self._name = value
        return self

    @property
    def has_init(self):
        return self._has_init

    @property
    def has_built_all(self):
        return self._has_built_all

    def get_cpu_list(self):
        return ['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin',
                'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense',
                'BatchMatMul', 'Gather', 'Print', 'InTopK', 'TopKV2',
                'SparseApplyMomentum']

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
        """Restore the weights.
        WARNING: Call only after built the graph.
        """
        # print '3', self.name
        if folder is not None:
            save_vars = self.get_save_var_dict()
            # print 'save var len', len(save_vars)
            # print save_vars
            if len(save_vars) > 0:
                Saver(folder, var_dict=save_vars,
                      fname=self.name).restore(sess)
        return self

    def restore_weights_from_ckpt(self, sess, file):
        if file is not None:
            save_vars = self.get_save_var_dict()
            if len(save_vars) > 0:
                tf.train.Saver(self.get_save_var_dict()).restore(sess, file)
        return self

    def restore_aux_from(self, sess, folder):
        """Restore the aux weights.
        WARNING: Call only after built the graph.
        """
        if folder is not None:
            aux_vars = self.get_aux_var_dict()
            if len(aux_vars) > 0:
                Saver(folder, var_dict=aux_vars,
                      fname=self.name + '.aux').restore(sess)
        return self

    def restore_weights_aux_from(self, sess, folder):
        self.restore_weights_from(sess, folder)
        self.restore_aux_from(sess, folder)
        return self

    @property
    def saver(self):
        if self.folder is None:
            raise Exception('Has not set save folder yet')
        if self._saver is None:
            save_vars = self.get_save_var_dict()
            if len(save_vars) > 0:
                self._saver = Saver(self.folder,
                                    var_dict=save_vars,
                                    fname=self.name)
        return self._saver

    @property
    def aux_saver(self):
        if self.folder is None:
            raise Exception('Has not set save folder yet')
        if self._aux_saver is None:
            aux_vars = self.get_aux_var_dict()
            if len(aux_vars) > 0:
                self._aux_saver = Saver(self.folder,
                                        var_dict=aux_vars,
                                        fname=self.name + '.aux')
        return self._aux_saver

    def save(self, sess, step=0):
        """Save model.

        Args:
            sess: tensorflow session
            step: number of steps
        """
        if self.folder is None:
            raise Exception('Has not set save folder yet')
        self.save_model_options()
        if self.saver is not None:
            self.saver.save(sess, global_step=step)
        if self.aux_saver is not None:
            self.aux_saver.save(sess, global_step=step)
        pass

    def save_model_options(self):
        if self.folder is None:
            raise Exception('Has not set save folder yet')
        self.save_options(self.folder, self.name)
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

    def build_loss(self, inp, output):
        """Build gradient graph. To be implemented by subclasses.

        """
        raise Exception('Not implemented')

    def build_optim(self, loss):
        raise Exception('Not implemented')

    @property
    def global_step(self):
        if self._global_step is None:
            self._global_step = tf.Variable(0.0)
            self.register_var('step', self._global_step)
        return self._global_step

    def build_eval(self):
        """Build nodes for evaluation/inference."""
        if self._has_built_all:
            raise Exception('Only call build_all or build_eval once.')
        self._has_built_all = True
        with tf.device(self.get_device_fn()):
            with tf.variable_scope(self.name):
                inp_var = self.build_input()
                output_var = self.build(inp_var)
        return self

    def build_loss_eval(self):
        """Build nodes for evaluation/inference."""
        if self._has_built_all:
            raise Exception('Only call build_all or build_eval once.')
        self._has_built_all = True
        with tf.device(self.get_device_fn()):
            with tf.variable_scope(self.name):
                inp_var = self.build_input()
                output_var = self.build(inp_var)
                gs = self.global_step
                loss_var = self.build_loss(inp_var, output_var)
        return self

    def build_all(self):
        """Build all nodes."""
        if self._has_built_all:
            raise Exception('Only call build_all or build_eval once.')
        self._has_built_all = True
        with tf.device(self.get_device_fn()):
            with tf.variable_scope(self.name):
                inp_var = self.build_input()
                output_var = self.build(inp_var)
                loss_var = self.build_loss(inp_var, output_var)
                train_step = self.build_optim(loss_var)
                self.register_var('train_step', train_step)
        return self

    def init(self, sess):
        if not self.has_built_all:
            raise Exception(
                'Need to call build_all or build_eval before init')
        self._has_init = True
        my_var_list = self.get_all_vars()
        sess.run(tf.initialize_variables(my_var_list))
        return self
    
    @property
    def var_dict(self):
        return self._var_dict

    def get_all_vars(self):
        var_list = tf.all_variables()
        my_var_list = []
        for v in var_list:
            if v.name.startswith(self.name):
                my_var_list.append(v)
                pass
            pass
        return my_var_list

    def get_save_var_dict(self):
        """Get a dictionary of variables to restore."""
        raise Exception('Not implemented')

    def get_aux_var_dict(self):
        all_vars = self.get_all_vars()
        all_save_vars = self.get_save_var_list_recursive()
        save_var_set = set(all_save_vars)
        print 'Save var length', len(save_var_set)
        print 'All var length', len(all_vars)
        aux_vars = {}
        for v in all_vars:
            if v not in save_var_set:
                aux_vars[v.name] = v
                print v.name, v
                pass
            pass
        return aux_vars

    def get_save_var_list_recursive(self):
        return self.get_save_var_dict().values()
    pass


class ContainerModel(Model):

    def __init__(self, name='model'):
        super(ContainerModel, self).__init__(name=name)
        self.sub_models = []
        pass

    def init_from_main(self):
        for m in self.sub_models:
            m.init_from_main()
            pass
        return super(ContainerModel, self).init_from_main()

    def add_sub_model(self, m):
        self.sub_models.append(m)
        pass

    def set_gpu(self, value):
        # For now all device is single.
        for m in self.sub_models:
            m.set_gpu(value)
            pass
        return super(ContainerModel, self).set_gpu(value)

    def set_folder(self, value):
        for m in self.sub_models:
            m.set_folder(value)
            pass
        return super(ContainerModel, self).set_folder(value)

    def save(self, sess, step=0):
        for m in self.sub_models:
            m.save(sess, step=step)
            pass
        return super(ContainerModel, self).save(sess, step=step)

    def save_model_options(self):
        for m in self.sub_models:
            m.save_model_options()
            pass
        return super(ContainerModel, self).save_model_options()

    def restore_options_from(self, folder):
        for m in self.sub_models:
            m.restore_options_from(folder)
            pass
        return super(ContainerModel, self).restore_options_from(folder)

    def restore_weights_from(self, sess, folder):
        # print '1'
        for m in self.sub_models:
            # print '2', m.name
            m.restore_weights_from(sess, folder)
            pass
        # print 'self'
        return super(ContainerModel, self).restore_weights_from(sess, folder)

    def restore_aux_from(self, sess, folder):
        for m in self.sub_models:
            m.restore_aux_from(sess, folder)
            pass
        return super(ContainerModel, self).restore_aux_from(sess, folder)

    def restore_weights_aux_from(self, sess, folder):
        for m in self.sub_models:
            m.restore_weights_aux_from(sess, folder)
            pass
        return super(ContainerModel, self).restore_weights_aux_from(
            sess, folder)

    def get_save_var_list_recursive(self):
        save_vars = []
        for m in self.sub_models:
            save_vars.extend(m.get_save_var_list_recursive())
        save_vars.extend(self.get_save_var_dict().values())
        return save_vars
