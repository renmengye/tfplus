import Queue
import threading
from data_provider import DataProvider
# import tfplus.utils


class Producer(threading.Thread):

    def __init__(self, q, data_provider):
        threading.Thread.__init__(self)
        self.q = q
        self.data_provider = data_provider

    def run(self):
        while True:
            self.q.put(self.data_provider.get_batch())
        pass
    pass


class ConcurrentDataProvider(DataProvider):

    def __init__(self, data_provider, max_queue_size=10, num_threads=5):
        """
        Data provider wrapper that supports concurrent data fetching.
        """
        super(ConcurrentDataProvider, self).__init__()
        self.q = Queue.Queue(maxsize=max_queue_size)
        self.fetchers = []
        for ii in xrange(num_threads):
            f = Producer(self.q, data_provider)
            f.start()
            self.fetchers.append(f)
        self.counter = 0
        pass

    def get_batch(self):
        if self.counter % 10 == 0:
            print 'Data queue size:', self.q.qsize()
            self.counter = 0
        batch = self.q.get()
        self.q.task_done()
        self.counter += 1
        return batch

    pass
