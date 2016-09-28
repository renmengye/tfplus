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
        self.log = tfplus.utils.logger.get()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):
        while not self.stopped():
            try:
                b = self.batch_iter.next()
                self.q.put(b)
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

    def __init__(self, batch_iter, max_queue_size=10, num_threads=5, log_queue=20):
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
        self.relaunch = True
        self.log_queue = log_queue
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
                self.log.info('Found one dead thread.', verbose=2)
                if self.relaunch:
                    self.log.info('Relaunch', verbose=2)
                    fnew = BatchProducer(self.q, self.batch_iter)
                    fnew.start()
                    self.fetchers.append(fnew)
            else:
                num_alive += 1
        if do_print:
            self.log.info('Number of alive threads: {}'.format(num_alive),
                          verbose=2)
            s = self.q.qsize()
            if s > self.max_queue_size / 3:
                self.log.info('Data queue size: {}'.format(s), verbose=2)
            else:
                self.log.warning('Data queue size: {}'.format(s))
        for dd in dead:
            self.fetchers.remove(dd)
        pass

    def next(self):
        self.scan(do_print=(self.counter % self.log_queue == 0))
        if self.counter % self.log_queue == 0:
            self.counter = 0
        batch = self.q.get()
        self.q.task_done()
        self.counter += 1
        while batch is None:
            self.log.info('Got an empty batch. Ending iteration.', verbose=2)
            self.relaunch = False
            try:
                batch = self.q.get(False)
                self.q.task_done()
                qempty = False
            except Queue.Empty:
                qempty = True
                pass

            if qempty:
                self.log.info('Queue empty. Scanning for alive thread.',
                              verbose=2)
                # Scan for alive thread.
                found_alive = False
                for ff in self.fetchers:
                    if ff.is_alive():
                        found_alive = True
                        break

                self.log.info('No alive thread found. Joining.', verbose=2)
                # If no alive thread, join all.
                if not found_alive:
                    for ff in self.fetchers:
                        ff.join()
                    raise StopIteration
            else:
                self.log.info('Got another batch from the queue.', verbose=2)
        return batch

    def reset(self):
        self.log.info('Resetting concurrent batch iter', verbose=2)
        self.log.info('Stopping all workers', verbose=2)
        for f in self.fetchers:
            f.stop()
        self.log.info('Cleaning queue', verbose=2)
        cleaner = BatchConsumer(self.q)
        cleaner.start()
        for f in self.fetchers:
            f.join()
        self.q.join()
        cleaner.stop()
        self.log.info('Resetting index', verbose=2)
        self.batch_iter.reset()
        self.log.info('Restarting workers', verbose=2)
        self.fetchers = []
        self.init_fetchers()
        self.relaunch = True
        pass
    pass

if __name__ == '__main__':
    from batch_iter import BatchIterator
    b = BatchIterator(100, batch_size=6, get_fn=None)
    cb = ConcurrentBatchIterator(b, max_queue_size=5, num_threads=3)
    for _batch in cb:
        log = tfplus.utils.logger.get()
        log.info(('Final out', _batch))
