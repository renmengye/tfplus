from data_provider import DataProvider


class SampleDataProvider(DataProvider):

    def __init__(self, data_provider):
        super(SampleDataProvider, self).__init__()
        self._data_provider = data_provider
        pass

    @property
    def data_provider(self):
        return self._data_provider

    def reject(self, idx):
        """
        Whether to reject a sample.
        """
        raise Exception('Not implemented')

    def get_size(self):
        return self.data_provider.get_size()

    def get_batch_idx(self, idx, **kwargs):
        print 'start'
        new_idx = self.reject(idx)
        print 'end'
        if len(new_idx) > 0:
            return self.data_provider.get_batch_idx(new_idx, **kwargs)
        else:
            return self.data_provider.get_batch_idx([new_idx[0]], **kwargs)
        pass
