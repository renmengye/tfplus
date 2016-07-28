import numpy as np
from plotter import Plotter
from listener import get_factory
import logger
import matplotlib.pyplot as plt

class ConfusionMatrixPlotter(Plotter):

    def __init__(self, num_cls=10, filename=None, name=None, cmap='gray'):
        super(ConfusionMatrixPlotter, self).__init__(
            filename=filename, name=name)
        self.num_cls = num_cls
        self.cmap = cmap
        self.log = logger.get()
        pass

    def plot(self, cls_out, cls_gt):
        cm = self.build_matrix(cls_out, cls_gt)
        plt.imshow(cm, cmap=self.cmap, interpolation='nearest')
        plt.colorbar()
        plt.tight_layout(pad=1.0, w_pad=0.0, h_pad=0.0)
        plt.savefig(self.filename, dpi=150)
        plt.close('all')
        pass

    def build_matrix(self, cls_out, cls_gt):
        cm = np.zeros([self.num_cls, self.num_cls], dtype='float32')
        for ii in xrange(cls_out.shape[0]):
            cm[cls_gt[ii], cls_out[ii]] += 1
        self.log.info('Unnormalized confusion matrix')
        self.log.info(cm)
        ss = cm.sum(axis=1).reshape([-1, 1])
        ss = ss + (ss == 0).astype('float32')
        cm = cm / ss
        self.log.info('Normalized confusion matrix')
        self.log.info(cm)
        return cm

    def listen(self, results):
        """Plot results.

        Args:
            images: [B, H, W] or [B, H, W, 3]
        """
        cls_out = results['class_out']
        cls_gt = results['class_gt']
        self.plot(cls_out, cls_gt)
        self.register()
    pass

get_factory().register('confusion', ConfusionMatrixPlotter)
