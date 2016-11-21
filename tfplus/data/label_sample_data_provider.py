from data_provider import DataProvider
import numpy as np
import tfplus


class LabelSampleDataProvider(DataProvider):

    def __init__(self, data_provider, stats_provider=None, mode='train'):
        super(LabelSampleDataProvider, self).__init__()
        self._data_provider = data_provider
        self._stats_provider = stats_provider
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
    def stats_provider(self):
        return self._stats_provider

    @property
    def data_provider(self):
        return self._data_provider

    def get_size(self):
        if self.mode == 'train':
            # Only iterating the keys (equalize the weights between different
            # classes).
            if self.stats_provider is None:
                return self._real_size * self.data_provider.get_size()
            else:
                return self.stats_provider.get_size()
        else:
            # Iterating the whole dataset.
            return self.data_provider.get_size()

    def get_batch_idx(self, idx, **kwargs):
        if self.mode == 'train':
            new_idx = []
            # self.log.info('Label IDX: {}'.format(idx))
            if self.stats_provider is None:
                label_ids = [ii % self._real_size for ii in idx]
            else:
                # print idx, self.stats_provider.get_size()
                stats_batch = self.stats_provider.get_batch_idx(idx)
                label_ids = []
                for ii in xrange(len(idx)):
                    label_ids.append(np.argmax(stats_batch['y_gt'][ii]))

            for ii in label_ids:
                data_group = self.data_provider.label_idx[ii]
                num_ids = len(data_group)
                kk = int(np.floor(self.rnd.uniform(0, num_ids)))
                new_idx.append(data_group[kk])
        else:
            new_idx = idx
        return self.data_provider.get_batch_idx(new_idx)
