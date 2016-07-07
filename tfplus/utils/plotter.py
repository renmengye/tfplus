from __future__ import division

from log_manager import LogManager

from listener import get_factory, Listener

import logger
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

_registry = {}


def register(name, plotter):
    _registry[name] = plotter
    pass


def get(name):
    return _registry[name]


class Plotter(Listener):

    def __init__(self, filename=None, name=None):
        super(Plotter, self).__init__()
        self._registered = False
        self._filename = filename
        self._name = name
        self._folder = os.path.dirname(filename)
        pass

    @property
    def name(self):
        return self._name

    @property
    def filename(self):
        return self._filename

    @property
    def registered(self):
        return self._registered

    @property
    def folder(self):
        return self._folder

    def listen(self):
        raise Exception('Not implemented')
        pass

    def register(self):
        if not self.registered:
            LogManager(self.folder).register(self.filename, 'image', self.name)
            self._registered = True
        pass
    pass


class ThumbnailPlotter(Plotter):

    def __init__(self, filename=None, name=None, cmap='Greys', max_num_col=9):
        super(ThumbnailPlotter, self).__init__(filename=filename, name=name)
        self._cmap = cmap
        self._max_num_col = max_num_col
        pass

    @property
    def cmap(self):
        return self._cmap

    @property
    def max_num_col(self):
        return self._max_num_col

    def plot(self, img):
        num_ex = 1
        num_items = img.shape[0]
        num_row, num_col, calc = self.calc_row_col(num_ex, num_items)

        f1, axarr = plt.subplots(num_row, num_col, figsize=(10, num_row))
        self.set_axis_off(axarr, num_row, num_col)

        for ii in xrange(num_items):
            row, col = calc(0, ii)
            x = img[ii]
            if num_col > 1:
                ax = axarr[row, col]
            else:
                ax = axarr[row]
            if x.shape[-1] == 3:
                x = x[:, :, [2, 1, 0]]
            elif x.shape[-1] == 1:
                x = x[:, :, 0]
            ax.imshow(x, cmap=self.cmap)
            ax.text(0, -0.5, '[{:.2g}, {:.2g}]'.format(x.min(), x.max()),
                    color=(0, 0, 0), size=8)

        plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(self.filename, dpi=150)
        plt.close('all')
        pass
    

    def calc_row_col(self, num_ex, num_items):
        num_rows_per_ex = int(np.ceil(num_items / self.max_num_col))
        if num_items > self.max_num_col:
            num_col = self.max_num_col
            num_row = num_rows_per_ex * num_ex
        else:
            num_row = num_ex
            num_col = num_items

        def calc(ii, jj):
            col = jj % self.max_num_col
            row = num_rows_per_ex * ii + int(jj / self.max_num_col)

            return row, col

        return num_row, num_col, calc

    def set_axis_off(self, axarr, num_row, num_col):
        for row in xrange(num_row):
            for col in xrange(num_col):
                if num_col > 1:
                    ax = axarr[row, col]
                else:
                    ax = axarr[row]
                ax.set_axis_off()
        pass

    def listen(self, results):
        """Plot results.

        Args:
            images: [B, H, W] or [B, H, W, 3]
        """
        img = results['images']
        self.plot(img)
        self.register()
    pass


get_factory().register('thumbnail', ThumbnailPlotter)
