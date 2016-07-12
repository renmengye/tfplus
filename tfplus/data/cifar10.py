from __future__ import division

import cPickle as pkl
import numpy as np
import os
import tfplus

tfplus.cmd_args.add('cifar10:dataset_folder', 'str',
                    '/ais/gobi4/mren/data/cifar10')


class CIFAR10DataProvider(tfplus.data.data_provider.DataProvider):

    def __init__(self, split='train', filename=None):
        super(CIFAR10DataProvider, self).__init__()
        self.log = tfplus.utils.logger.get()
        if split is None:
            self.split = 'train'
        else:
            self.split = split
        self.log.info('Data split: {}'.format(self.split))
        self.filename = filename
        self._images = None
        self._labels = None
        self.register_option('cifar10:dataset_folder')
        pass

    def init_data(self):
        if self.split == 'train':
            self._images = np.zeros([50000, 32, 32, 3], dtype='uint8')
            self._labels = np.zeros([50000], dtype='int')
            for batch in xrange(5):
                fname = os.path.join(self.get_option(
                    'cifar10:dataset_folder'), 'data_batch_{}'.format(
                    batch + 1))
                start = batch * 10000
                end = (batch + 1) * 10000
                with open(fname, 'rb') as fo:
                    _data = pkl.load(fo)
                    self._images[start: end] = _data['data'].reshape(
                        [10000, 3, 32, 32]).transpose([0, 2, 3, 1])
                    self._labels[start: end] = np.array(_data['labels'])
        elif self.split == 'test':
            fname = os.path.join(self.get_option(
                'cifar10:dataset_folder'), 'test_batch')
            with open(fname, 'rb') as fo:
                _data = pkl.load(fo)
                self._images = _data['data'].reshape(
                    [10000, 3, 32, 32]).transpose([0, 2, 3, 1])
                print _data.keys()
                self._labels = np.array(_data['labels'])
            pass
        else:
            raise Exception('Unknown split: {}'.format(self.split))
        pass

    def get_size(self):
        if self.split == 'train':
            return 50000
        elif self.split == 'test':
            return 10000
        else:
            raise Exception('Unknown split {}'.format(self.split))

    def get_batch(self, idx, **kwargs):
        if self._images is None:
            self.init_data()
        labels = self._labels[idx]
        y_gt = np.zeros([len(idx), 10], dtype='float32')
        y_gt[np.arange(len(idx)), labels] = 1.0
        results = {
            'x': (self._images[idx] / 255).astype('float32'),
            'y_gt': y_gt
        }
        return results

tfplus.data.data_provider.get_factory().register('cifar10', CIFAR10DataProvider)


if __name__ == '__main__':
    print tfplus.data.data_provider.create_from_main('cifar10').get_batch(
        np.arange(5))
    pass
