from listener import get_factory, Listener
import time_series_logger as ts_logger


class CSVListener(Listener):

    def __init__(self, name=None, var_name=None, label=None):
        super(CSVListener, self).__init__()
        self.var_name = var_name
        self.label = label
        self.logger = ts_logger.get(name)
        pass

    def listen(self, results):
        self.logger.add_one(int(results['step']), results[
                            self.var_name], label=self.label)
        pass

    pass

get_factory().register('csv', CSVListener)
