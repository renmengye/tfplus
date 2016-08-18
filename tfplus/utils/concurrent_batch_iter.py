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
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while not self.stopped():
            try:
                self.q.put(self.batch_iter.next())
            except StopIteration:
                self.q.put(None)
                break
        pass
    pass


class BatchConsumer(threading.Thread):

    def __init__(self, q):
        threading.Thread.__init__(self)
        self.q = q
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while not self.stopped():
            try:
                self.q.get(False)
                self.q.task_done()
            except Queue.Empty:
                pass
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
        self.batch_iter = batch_iter
        self.fetchers = []
        self.init_fetchers()
        self.counter = 0
        pass

    def init_fetchers(self):
        for ii in xrange(self.num_threads):
            f = BatchProducer(self.q, self.batch_iter)
            f.start()
            self.fetchers.append(f)
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
        self.q.task_done()
        self.counter += 1
        if batch is None:
            raise StopIteration
        return batch

    def reset(self):
        self.log.info('Resetting concurrent batch iter')
        self.log.info('Stopping all workers')
        for f in self.fetchers:
            f.stop()
        self.log.info('Cleaning queue')
        cleaner = BatchConsumer(self.q)
        cleaner.start()
        for f in self.fetchers:
            f.join()
        self.q.join()
        cleaner.stop()
        self.log.info('Resetting index')
        self.batch_iter.reset()
        self.log.info('Restarting workers')
        self.fetchers = []
        self.init_fetchers()
        pass
    pass
