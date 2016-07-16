import numpy as np
import os
import time

from tfplus.utils import cmd_args, logger, listener, OptionBase, Factory
from tfplus.utils import plotter

cmd_args.add('save_ckpt', 'bool', False)

_factory = None


def get_factory():
    global _factory
    if _factory is None:
        _factory = Factory()
        pass
    return _factory


def create(_clsname, **kwargs):
    return get_factory().create(_clsname, **kwargs)


def create_from_main(_clsname, **kwargs):
    return get_factory().create_from_main(_clsname, **kwargs)


class EmptyRunner(OptionBase):

    def __init__(self):
        super(EmptyRunner, self).__init__()
        self._name = 'default'
        self._interval = 1
        self._offset = 0
        self._experiment = None
        pass

    def run_step(self):
        pass

    def finalize(self):
        pass

    def get_name(self):
        return self._name

    def set_name(self, value):
        self._name = value
        return self

    @property
    def name(self):
        return self._name

    def get_interval(self):
        return self._interval

    def set_interval(self, value):
        self._interval = value
        return self

    @property
    def interval(self):
        return self._interval

    def get_offset(self):
        return self._offset

    def set_offset(self, value):
        self._offset = value
        return self

    @property
    def offset(self):
        return self._offset

    def get_experiment(self):
        return self._experiment

    def set_experiment(self, value):
        self._experiment = value
        return self

    @property
    def experiment(self):
        return self._experiment


class SessionRunner(EmptyRunner):

    def __init__(self):
        super(SessionRunner, self).__init__()
        self._sess = None
        self._model = None
        pass

    def get_session(self):
        return self._sess

    def set_session(self, value):
        self._sess = value
        return self

    def get_model(self):
        return self._model

    def set_model(self, value):
        self._model = value
        return self

    @property
    def session(self):
        return self._sess

    @property
    def model(self):
        return self._model


class SaverRunner(SessionRunner):

    def __init__(self):
        super(SaverRunner, self).__init__()
        self._log = logger.get()
        self.register_option('save_ckpt')
        pass

    def get_save_ckpt(self):
        return self.get_option('save_ckpt')

    def set_save_ckpt(self, value):
        return self.set_option('save_ckpt', value)

    def run_step(self):
        step = self.get_session().run(self.model.get_var('step'))
        step = int(step)
        if self.get_option('save_ckpt'):
            self._log.info('Saving checkpoint')
            self.model.save(self.get_session(), step=step)
            pass
        else:
            self._log.warning(
                'Saving is turned off. Use --save_ckpt flag to save.')
            pass
        pass
    pass

get_factory().register('saver', SaverRunner)


class RestorerRunner(SessionRunner):

    def __init__(self, folder=None):
        super(RestorerRunner, self).__init__()
        self._log = logger.get()
        self.folder = folder
        pass

    def set_folder(self, value):
        self.folder = value
        return self

    def run_step(self):
        step = self.get_session().run(self.model.get_var('step'))
        step = int(step)
        self._log.info('Restoring checkpoint')
        self.model.restore_weights_from(self.get_session(), self.folder)
        pass
    pass

get_factory().register('restorer', RestorerRunner)


class BasicRunner(SessionRunner):

    def __init__(self):
        super(BasicRunner, self).__init__()
        self._step = 0
        self._data_provider = None
        self._phase_train = True
        self._outputs = []
        self._current_batch = {}
        self._log = logger.get()
        self._preprocessor = lambda x: x
        self._listeners = []
        pass

    @property
    def listeners(self):
        return self._listeners

    def write_log(self, results):
        for listener in self.listeners:
            listener.listen(results)
            pass

    def add_listener(self, listener):
        self.listeners.append(listener)
        return self

    def add_csv_listener(self, name, var_name, label=None):
        if var_name not in self.outputs and var_name != 'step_time':
            raise Exception(
                'Runner "{}": variable "{}" is not in output list.'.format(
                    self.name, var_name))
        self.listeners.append(listener.get_factory().create(
            'csv', name=name, var_name=var_name, label=label))
        return self

    def add_plot_listener(self, name, mapping):
        for var_name in mapping.iterkeys():
            if var_name not in self.outputs and var_name != 'step_time':
                raise Exception(
                    'Runner "{}": variable "{}" is not in output list.'.format(
                        self.name, var_name))
        self.listeners.append(listener.AdapterListener(
            mapping=mapping,
            listener=plotter.get(name)))
        return self

    def add_cmd_listener(self, name, var_name):
        if var_name not in self.outputs and var_name != 'step_time':
            raise Exception(
                'Runner "{}": variable "{}" is not in output list.'.format(
                    self.name, var_name))
        self.listeners.append(listener.get_factory().create(
            'cmd', name=name, var_name=var_name))
        return self

    def finalize(self):
        for key, value in self.loggers.items():
            value.close()
            pass
        pass

    @property
    def log(self):
        return self._log

    @property
    def phase_train(self):
        return self._phase_train
    pass

    def get_phase_train(self):
        return self._phase_train

    def set_phase_train(self, value):
        self._phase_train = value
        return self

    @property
    def data_provider(self):
        return self._data_provider

    def get_data_provider(self):
        return self._data_provider

    def set_data_provider(self, value):
        self._data_provider = value
        return self

    @property
    def outputs(self):
        return self._outputs

    def get_outputs(self):
        return self._outputs

    def set_outputs(self, value):
        self._outputs = value
        if 'step' not in self._outputs:
            self._outputs.append('step')
        return self

    @property
    def preprocessor(self):
        return self._preprocessor

    def set_preprocessor(self, value):
        self._preprocessor = value
        return self

    @property
    def step(self):
        return self._step

    @property
    def current_batch(self):
        return self._current_batch

    def _run_step(self, inp):
        """Train step"""
        self._current_batch = inp
        bat_sz_total = 0
        results = {}

        feed_dict = self.get_feed_dict(inp)
        start_time = time.time()
        r = self.run_model(inp)
        step_time = (time.time() - start_time) * 1000
        r['step_time'] = step_time
        self._step = int(r['step'])
        return r

    def get_feed_dict(self, inp):
        inp = self._preprocessor(inp)
        feed_dict = {self.model.get_input_var('phase_train'): self.phase_train}
        set_var = False
        for key in inp.iterkeys():
            if self.model.has_input_var(key):
                feed_dict[self.model.get_input_var(key)] = inp[key]
                pass
            else:
                self.log.warning(
                    'Ignoring input variable "{}".'.format(key))
                set_var = True
                pass
        if set_var:
            # Set a subset of variables.
            if self.data_provider.variables is None:
                self.data_provider.set_variables(
                    self.model.get_all_input_vars().keys())
                self.log.warning('Setting input variable list: {}'.format(
                    self.data_provider.variables))
                pass
            pass
        return feed_dict

    def run_model(self, inp):
        feed_dict = self.get_feed_dict(inp)
        symbol_list = [self.model.get_var(r) for r in self.outputs]
        results = self.session.run(symbol_list, feed_dict=feed_dict)
        results_dict = {}
        for rr, name in zip(results, self.outputs):
            results_dict[name] = rr
        return results_dict

    def run_step(self):
        try:
            inp = self.data_provider.get_batch()
        except StopIteration:
            return False
        results = self._run_step(inp)
        self.write_log(results)
        return True

    pass

get_factory().register('basic', BasicRunner)


class AverageRunner(BasicRunner):

    def __init__(self):
        super(AverageRunner, self).__init__()
        self._num_batch = 1
        pass

    @property
    def num_batch(self):
        return self._num_batch

    def get_num_batch(self):
        return self._num_batch

    def set_num_batch(self, value):
        self._num_batch = value
        return self

    def run_step(self):
        bat_sz_total = 0
        results = {}

        # Initialize values.
        if len(self.outputs) == 0:
            self.log.warning(
                'Empty outputs list for runner "{}"'.format(self.name))
        for key in self.outputs:
            results[key] = 0.0
            pass
        results['step_time'] = 0.0

        # Run each batch.
        for bb in xrange(self.num_batch):
            try:
                inp = self.data_provider.get_batch()
            except StopIteration:
                return False
            _results = self._run_step(inp)
            bat_sz = inp[inp.keys()[0]].shape[0]
            bat_sz_total += bat_sz
            for key in _results.iterkeys():
                if _results[key] is not None:
                    results[key] += _results[key] * bat_sz
                pass
            pass

        # Average out all batches.
        for key in results.iterkeys():
            results[key] = results[key] / bat_sz_total
            pass

        # Do not average steps.
        results['step'] = self.step
        self.write_log(results)

        return True

get_factory().register('average', AverageRunner)
