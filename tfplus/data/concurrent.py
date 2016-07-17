from __future__ import division

import Queue
import threading
import tfplus

from data_provider import DataProvider
# import tfplus.utils


class Producer(threading.Thread):

    def __init__(self, q, data_provider):
        threading.Thread.__init__(self)
        self.q = q
        self.data_provider = data_provider

    def run(self):
        while True:
            try:
                self.q.put(self.data_provider.get_batch())
            except StopIteration:
                self.q.put(None)
                break
        pass
    pass


class ConcurrentDataProvider(DataProvider):

    def __init__(self, data_provider, max_queue_size=10, num_threads=5):
        """
        Data provider wrapper that supports concurrent data fetching.
        """
        super(ConcurrentDataProvider, self).__init__()
        self.max_queue_size = max_queue_size
        self.num_threads = num_threads
        self.q = Queue.Queue(maxsize=max_queue_size)
        self.log = tfplus.utils.logger.get()
        self.fetchers = []
        for ii in xrange(num_threads):
            f = Producer(self.q, data_provider)
            f.start()
            self.fetchers.append(f)
        self.data_provider = data_provider
        self.counter = 0
        pass

    def scan(self):
        dead = []
        num_alive = 0
        for ff in self.fetchers:
            if not ff.is_alive():
                dead.append(ff)
                self.log.info('Found one dead thread. Relaunching.')
                fnew = Producer(self.q, self.data_provider)
                fnew.start()
                self.fetchers.append(fnew)
            else:
                num_alive += 1
        self.log.info('Number of alive threads: {}'.format(num_alive))
        for dd in dead:
            self.fetchers.remove(dd)
        pass

    def get_batch(self):
        if self.counter % 10 == 0:
            s = self.q.qsize()
            if s > self.max_queue_size / 3:
                self.log.info('Data queue size: {}'.format(s))
            else:
                self.log.warning('Data queue size: {}'.format(s))
            self.scan()
            self.counter = 0
        batch = self.q.get()
        if batch is None:
            raise StopIteration
        self.q.task_done()
        self.counter += 1

        return batch
    pass
