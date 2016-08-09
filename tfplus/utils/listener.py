from tfplus.utils import Factory

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


class Listener(object):

    def listen(self, results):
        pass

    def finalize(self):
        pass

    def stage(self):
        pass
    pass


class AdapterListener(Listener):

    def __init__(self, mapping=None, listener=None):
        self._mapping = mapping
        self._listener = listener
        pass

    def listen(self, results):
        results2 = {}
        for inp, out in self._mapping.items():
            results2[out] = results[inp]
            pass
        return self._listener.listen(results2)

get_factory().register('adapter', AdapterListener)
