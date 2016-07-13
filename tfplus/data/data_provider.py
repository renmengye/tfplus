import os

from tfplus.utils import cmd_args, OptionBase, Factory, BatchIterator

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


class DataProvider(OptionBase):

    def __init__(self):
        super(DataProvider, self).__init__()
        self._variables = None
        self._iterator = None
        pass

    @property
    def iterator(self):
        return self._iterator

    @property
    def variables(self):
        return self._variables

    def set_variables(self, value):
        self._variables = value
        return self

    def get_size(self):
        """Get number of examples."""
        raise Exception('Not implemented')

    def get_batch(self, **kwargs):
        """Get a batch of data.
        """
        return self.get_batch_idx(self.iterator.next(), **kwargs)

    def get_batch_idx(self, idx, **kwargs):
        """Get a batch of data.
        """
        raise Exception('Not implemented')

    def set_iter(self, batch_size=1, progress_bar=False, cycle=False,
                 shuffle=True, stagnant=False):
        """Get a batch iterator of data.

        Args:
            See BatchIterator.
        """
        self._iterator = BatchIterator(
            self.get_size(), batch_size=batch_size,
            progress_bar=progress_bar,
            cycle=cycle, shuffle=shuffle, stagnant=stagnant)
        return self
    pass
