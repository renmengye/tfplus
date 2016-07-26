from data_provider import DataProvider
import numpy as np
import tfplus


class LabelSampleDataProvider(DataProvider):

    def __init__(self, data_provider, mode='train'):
        super(LabelSampleDataProvider, self).__init__()
        self._data_provider = data_provider
        self._rnd = np.random.RandomState(2)
        self._mode = mode
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
        # Only iterating the keys (equalize the weights between different
        # classes)
        return len(self.data_provider.label_idx.keys())

    def get_batch_idx(self, idx, **kwargs):
        if self.mode == 'train':
            new_idx = []
            # self.log.info('Label IDX: {}'.format(idx))
            for ii in idx:
                data_group = self.data_provider.label_idx[ii]
                num_ids = len(data_group)
                kk = int(np.floor(self.rnd.uniform(0, num_ids)))
                new_idx.append(data_group[kk])
            # self.log.info('Ex IDX: {}'.format(new_idx))
        else:
            new_idx = idx
        return self.data_provider.get_batch_idx(new_idx)
