"""
Usage:
data_pro = data_provider.create_from_main('mnist')
model = model.get_factory().create_from_main('test_model')
expr = experiment.get_factory().create_from_main('exp1')
                 # .set_data_provider(data_pro)
                 .set_model(model)
                 .add_runner('train', Runner())
                 .add_runner('trainval', Runner())
                 .add_runner('valid', Runner())
                 .add_logger('train', 'loss')
                 .add_logger('train', 'acc')
                 .add_logger('valid', 'acc')
                 .run()
"""

from __future__ import division

import os
import tensorflow as tf
import sys

from tfplus.utils import cmd_args, Factory, OptionBase, Saver, logger
from tfplus.utils import time_series_logger as ts_logger
from tfplus.utils import plotter, listener, LogManager

cmd_args.add('num_steps', 'int', 500000)

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


class Experiment(OptionBase):
    """
    Experiment abstract definition.
    """

    def __init__(self, sess=None, model=None):
        super(Experiment, self).__init__()
        self.log = logger.get()
        self.log.log_args()
        self._sess = sess
        self._model = model
        pass

    @property
    def session(self):
        return self._sess

    def set_session(self, value):
        self._sess = value
        return self

    @property
    def model(self):
        return self._model

    def set_model(self, value):
        self._model = value
        return self

    def run(self):
        raise Exception('Not implemented')

    pass


class EvalExperiment(Experiment):

    def __init__(self, sess=None, model=None):
        super(EvalExperiment, self).__init__(sess=sess, model=model)
        pass

    pass


class TrainExperiment(Experiment):

    def __init__(self, sess=None, model=None):
        super(TrainExperiment, self).__init__(sess=sess, model=model)
        self.runners = {}
        self.register_option('num_steps')
        # self.log_manager = None
        self._logs_folder = '../logs'
        self._localhost = 'http://localhost'
        self.ts_loggers = {}
        self._preprocessor = lambda x: x
        pass

    @property
    def logs_folder(self):
        if not os.path.exists(self._logs_folder):
            os.makedirs(self._logs_folder)
            pass
        return self._logs_folder

    def set_logs_folder(self, value):
        self._logs_folder = value
        return self

    @property
    def localhost(self):
        return self._localhost

    def set_localhost(self, value):
        self._localhost = value
        return self

    @property
    def preprocessor(self):
        return self._preprocessor

    def set_preprocessor(self, value):
        self._preprocessor = value

    def add_runner(self, runner):
        self.runners[runner.name] = runner
        runner.set_session(self.session).set_model(
            self.model).set_experiment(self).set_preprocessor(self.preprocessor)
        return self

    def add_csv_output(self, name, labels):
        if name not in self.ts_loggers:
            self.ts_loggers[name] = ts_logger.register(
                os.path.join(self.logs_folder, name + '.csv'), labels, name,
                buffer_size=1)
        else:
            raise Exception('TS logger "{}" already registered'.format(name))
        return self

    def add_plot_output(self, name, typ, **kwargs):
        plotter.register(name, listener.create(
            typ, name=name,
            filename=os.path.join(self.logs_folder, name + '.png'),
            **kwargs))
        return self

    def restore_logs(self, logs_folder):
        return self

    def run(self):
        if 'train' not in self.runners:
            raise Exception('Need to specify runner "train"')

        self.url = os.path.join(self.localhost, 'deep-dashboard') + '?id=' + \
            self.logs_folder.split('/')[-1]

        # Register model hyperparams.
        self.model.save_options(self.logs_folder, 'model')
        log_manager = LogManager(self.logs_folder)
        log_manager.register(
            'model.yaml', 'plain', 'Model Hyperparameters')
        cmd_fname = os.path.join(self.logs_folder, 'cmd.log')
        with open(cmd_fname, 'w') as f:
            f.write(' '.join(sys.argv))
        log_manager.register(
            'cmd.log', 'plain', 'Command-line Arguments')

        # Counters
        count = 0
        step = self.runners['train'].step

        while step < self.get_option('num_steps'):

            # Runners
            for name, runner in self.runners.items():
                if count % runner.interval == 0:
                    if runner.interval > 1:
                        self.log.info('Runner "{}"'.format(name))
                        pass
                    runner.run_step()
                    pass
                pass

            # Dashboard reminder
            if count % 10 == 0:
                self.log.info('Dashboard {}'.format(self.url))
                pass

            count += 1
            step = self.runners['train'].step
            pass

        for runner in self.runners.itervalues():
            runner.finalize()
            pass

        self.session.close()

        for logger in self.loggers.itervalues():
            logger.close()
            pass
        pass
    pass

get_factory().register('train', TrainExperiment)
