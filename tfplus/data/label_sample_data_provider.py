from data_provider import DataProvider
import numpy as np
import tfplus


class LabelSampleDataProvider(DataProvider):

    def __init__(self, data_provider, mode='train'):
        super(LabelSampleDataProvider, self).__init__()
        self._data_provider = data_provider
        self._rnd = np.random.RandomState(2)
        self._mode = mode
        self._real_size = len(self.data_provider.label_idx.keys())
        self.log = tfplus.utils.logger.get()
        pass

    @property
    def mode(self):
        return self._mode

    @property
    def rnd(self):
        return self._rnd

    @property
    def data_provider(self):
        return self._data_provider

    def get_size(self):
        if self.mode == 'train':
            # Only iterating the keys (equalize the weights between different
            # classes).
            return self._real_size * 10000
        else:
            # Iterating the whole dataset.
            return self.data_provider.get_size()

    def get_batch_idx(self, idx, **kwargs):
        if self.mode == 'train':
            new_idx = []
            # self.log.info('Label IDX: {}'.format(idx))
            for ii in idx:
                ii_ = ii % self._real_size
                data_group = self.data_provider.label_idx[ii_]
                num_ids = len(data_group)
                kk = int(np.floor(self.rnd.uniform(0, num_ids)))
                new_idx.append(data_group[kk])
            # self.log.info('Ex IDX: {}'.format(new_idx))
        else:
            # self.log.info('Eval mode idx: {}'.format(idx))
            new_idx = idx
        return self.data_provider.get_batch_idx(new_idx)
