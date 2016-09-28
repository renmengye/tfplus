from __future__ import division
from tfplus.utils import Listener
from tfplus.utils import LogManager
from tfplus.utils import logger
import datetime
import os
import numpy as np


class AccuracyListener(Listener):

    def __init__(self, top_k=1, filename=None, label=None):
        self.top_k = top_k
        self.correct = 0
        self.count = 0
        self.log = logger.get()
        self.filename = filename
        self.label = label
        self.registered = False
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('step,time,{}\n'.format(label))
        pass

    def listen(self, results):
        score_out = results['score_out']
        y_gt = results['y_gt']
        sort_idx = np.argsort(score_out, axis=-1)
        idx_gt = np.argmax(y_gt, axis=-1)
        correct = 0
        count = 0
        for kk, ii in enumerate(idx_gt):
            sort_idx_ = sort_idx[kk][::-1]
            for jj in sort_idx_[:self.top_k]:
                if ii == jj:
                    correct += 1
                    break
            count += 1
        # self.log.info('Correct {}/{}'.format(correct, count))
        self.correct += correct
        self.count += count
        self.step = int(results['step'])
        # self.log.info('Step {}'.format(self.step))
        pass

    def stage(self):
        acc = self.correct / self.count
        self.log.info('Top-{} Accuracy: {:.4f}'.format(self.top_k, acc))
        self.correct = 0
        self.count = 0
        if self.filename is not None:
            t = datetime.datetime.utcnow()
            with open(self.filename, 'a') as f:
                f.write('{:d},{},{}\n'.format(self.step, t.isoformat(), acc))
        if not self.registered:
            LogManager(os.path.dirname(self.filename)).register(
                os.path.basename(self.filename), 'csv',
                'Top {} Accuracy (Valid)'.format(self.top_k))
            self.registered = True
        pass
