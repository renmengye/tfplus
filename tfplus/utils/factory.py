"""
A simple factory.
"""

import logger


class Factory(object):

    def __init__(self):
        self._reg = {}
        self.log = logger.get()
        pass

    def register(self, name, cls):
        if name not in self._reg:
            self.log.info('Registering class "{}"'.format(name), verbose=2)
            self._reg[name] = cls
        else:
            raise Exception('Class "{}" already registered')
        pass

    def create(self, _clsname, **kwargs):
        if _clsname in self._reg:
            return self._reg[_clsname](**kwargs)
        else:
            raise Exception('Class "{}" not registered'.format(_clsname))

    def create_from_main(self, _clsname, **kwargs):
        if _clsname in self._reg:
            return self._reg[_clsname](**kwargs).init_from_main()
        else:
            raise Exception('Class "{}" not registered'.format(_clsname))
