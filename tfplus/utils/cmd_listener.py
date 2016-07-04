from listener import get_factory, Listener
import logger
import numpy as np

class CmdListener(Listener):

    def __init__(self, name=None, var_name=None):
        super(CmdListener, self).__init__()
        self.var_name = var_name
        self.name = name
        self.log = logger.get()
        pass

    def listen(self, results):
        typ = type(results[self.var_name])
        if typ == float or typ == np.float64 or typ == np.float32:
            typstr = '{:.4f}'
        else:
            typstr = '{}'
        self.log.info(
            ('{} ' + typstr).format(self.name, results[self.var_name]))
        pass
    pass

get_factory().register('cmd', CmdListener)
