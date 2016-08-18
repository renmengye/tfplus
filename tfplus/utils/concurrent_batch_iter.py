from __future__ import division

import Queue
import threading
import tfplus

from batch_iter import IBatchIterator, BatchIterator


class BatchProducer(threading.Thread):

    def __init__(self, q, batch_iter):
        threading.Thread.__init__(self)
        self.q = q
        self.batch_iter = batch_iter

    def run(self):
        while True:
            try:
                self.q.put(self.batch_iter.next())
            except StopIteration:
                self.q.put(None)
                break
        pass
    pass


class ConcurrentBatchIterator(IBatchIterator):

    def __init__(self, batch_iter, max_queue_size=10, num_threads=5):
        """
        Data provider wrapper that supports concurrent data fetching.
        """
        super(ConcurrentBatchIterator, self).__init__()
        self.max_queue_size = max_queue_size
        self.num_threads = num_threads
        self.q = Queue.Queue(maxsize=max_queue_size)
        self.log = tfplus.utils.logger.get()
        self.fetchers = []
        for ii in xrange(num_threads):
            f = BatchProducer(self.q, batch_iter)
            f.start()
            self.fetchers.append(f)
        self.batch_iter = batch_iter
        self.counter = 0
        pass

    def scan(self, do_print=False):
        dead = []
        num_alive = 0
        for ff in self.fetchers:
            if not ff.is_alive():
                dead.append(ff)
                self.log.info('Found one dead thread. Relaunching.')
                fnew = BatchProducer(self.q, self.batch_iter)
                fnew.start()
                self.fetchers.append(fnew)
            else:
                num_alive += 1
        if do_print:
            self.log.info('Number of alive threads: {}'.format(num_alive))
            s = self.q.qsize()
            if s > self.max_queue_size / 3:
                self.log.info('Data queue size: {}'.format(s))
            else:
                self.log.warning('Data queue size: {}'.format(s))
        for dd in dead:
            self.fetchers.remove(dd)
        pass

    def next(self):
        self.scan(self.counter % 20 == 0)
        if self.counter % 20 == 0:
            self.counter = 0
        batch = self.q.get()
        if batch is None:
            raise StopIteration
        self.q.task_done()
        self.counter += 1
        return batch

    def reset(self):
        self.batch_iter.reset()
        pass
    pass
