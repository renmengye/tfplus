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
        pass

    def get_size(self):
        """Get number of examples."""
        raise Exception('Not implemented')

    def get_batch(self, idx, **kwargs):
        """Get a batch of data.

        Args:
            idx: numpy.ndarray of dtype int. 0-based index of data entries.
        """
        raise Exception('Not implemented')

    def get_iter(self, batch_size=1, progress_bar=False, cycle=False, shuffle=False, stagnant=False):
        """Get a batch iterator of data.

        Args:
            See BatchIterator.
        """
        return BatchIterator(self.get_size(), batch_size=batch_size,
                             progress_bar=progress_bar, get_fn=self.get_batch,
                             cycle=cycle, shuffle=shuffle, stagnant=stagnant)
    pass
