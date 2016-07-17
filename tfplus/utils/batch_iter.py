"""
A batch iterator.

Usage:
    for idx in BatchIterator(num=1000, batch_size=25):
        inp_batch = inp_all[idx]
        labels_batch = labels_all[idx]
        train(inp_batch, labels_batch)
"""

import numpy as np
import progress_bar as pb
import threading


class BatchIterator(object):

    def __init__(self, num, batch_size=1, progress_bar=False, get_fn=None, cycle=False, shuffle=True, stagnant=False):
        """Construct a batch iterator.

        Args:
            data: numpy.ndarray, (N, D), N is the number of examples, D is the
            feature dimension.
            labels: numpy.ndarray, (N), N is the number of examples.
            batch_size: int, batch size.
        """

        self._num = num
        self._batch_size = batch_size
        self._step = 0
        self._num_steps = int(np.ceil(self._num / float(batch_size)))
        self._pb = None
        self._variables = None
        self._get_fn = get_fn
        self.get_fn = get_fn
        self._cycle = cycle
        self._shuffle_idx = np.arange(self._num)
        self._shuffle = shuffle
        self._random = np.random.RandomState(2)
        self._shuffle_flag = shuffle
        self._stagnant = stagnant
        if progress_bar:
            self._pb = pb.get(self._num_steps)
            pass
        self._mutex = threading.Lock()
        pass

    def __iter__(self):
        """Get iterable."""
        return self

    def __len__(self):
        """Get iterable length."""
        return self._num_steps

    @property
    def variables(self):
        return self._variables

    def set_variables(self, variables):
        self._variables = variables

        def get_fn(idx):
            return self._get_fn(idx, variables=variables)
        self.get_fn = get_fn
        return self

    def reset(self):
        self._step = 0

    def next(self):
        """Iterate next element."""
        self._mutex.acquire()
        try:
            # Shuffle data.
            if self._shuffle_flag:
                self._random.shuffle(self._shuffle_idx)
                self._shuffle_flag = False

            # Read/write of self._step stay in a thread-safe block.
            if not self._cycle:
                if self._step >= self._num_steps:
                    raise StopIteration()

            # Calc start/end based on current step.
            start = self._batch_size * self._step
            end = self._batch_size * (self._step + 1)

            # Progress bar.
            if self._pb is not None:
                self._pb.increment()

            # Increment step.
            if not self._stagnant:
                self._step += 1
        finally:
            self._mutex.release()

        if not self._cycle:
            end = min(self._num, end)
            idx = np.arange(start, end)
            if self.get_fn is not None:
                return self.get_fn(idx)
            else:
                return idx
        else:
            start = start % self._num
            end = end % self._num
            if end > start:
                idx = np.arange(start, end)
                idx = self._shuffle_idx[idx]
            else:
                idx = np.array(range(start, self._num) + range(0, end))
                idx = self._shuffle_idx[idx]
                # Shuffle every cycle.
                if self._shuffle:
                    self._shuffle_flag = True
            if self.get_fn is not None:
                return self.get_fn(idx)
            else:
                return idx
        pass

if __name__ == '__main__':
    for ii in BatchIterator(400, batch_size=32, progress_bar=True, 
        get_fn=lambda x: x, cycle=True, shuffle=False):
        print ii
