from __future__ import division

import cv2
import numpy as np
import threading
import data_provider
from data_provider import DataProvider


class ListImageDataProvider(DataProvider):
    """
    Pass in a list of image file path in a plain text file.
    """

    def __init__(self, fname, inp_height=None, inp_width=None):
        super(ListImageDataProvider, self).__init__()
        self._ids = None
        self._fname = fname
        self._id_lock = threading.Lock()
        self._inp_height = inp_height
        self._inp_width = inp_width
        pass

    @property
    def fname(self):
        return self._fname

    @property
    def ids(self):
        if self._ids is None:
            self._id_lock.acquire()
            try:
                with open(self.fname, 'r') as f:
                    ids = f.readlines()
                ids = [ii.strip('\n') for ii in ids]
                self._ids = ids
            finally:
                self._id_lock.release()
            pass
        return self._ids

    @property
    def inp_height(self):
        return self._inp_height

    @property
    def inp_width(self):
        return self._inp_width

    def get_size(self):
        return len(self.ids)

    def get_batch_idx(self, idx):
        hh = self.inp_height
        ww = self.inp_width
        x = np.zeros([len(idx), hh, ww, 3], dtype='float32')
        orig_height = []
        orig_width = []
        ids = []
        for kk, ii in enumerate(idx):
            fname = self.ids[ii]
            ids.append('{:06}'.format(ii))
            x_ = cv2.imread(fname).astype('float32') / 255
            x[kk] = cv2.resize(
                x_, (self.inp_width, self.inp_height),
                interpolation=cv2.INTER_CUBIC)
            orig_height.append(x_.shape[0])
            orig_width.append(x_.shape[1])
            pass
        return {
            'x': x,
            'orig_height': np.array(orig_height),
            'orig_width': np.array(orig_width),
            'id': ids
        }
    pass

data_provider.get_factory().register('list_img', ListImageDataProvider)
