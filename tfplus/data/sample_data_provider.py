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

    def get_batch_idx(self, idx, **kwargs):
        new_idx = self.reject(idx)
        if len(new_idx) > 0:
            return self.data_provider.get_batch_idx(idx, **kwargs)
        else:
            return self.data_provider.get_batch_idx([idx[0]], **kwargs)
        pass
