from tfplus.data import data_provider
import gzip
import os
import urllib

import numpy
from tfplus.utils import logger, cmd_args

SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'
VALIDATION_SIZE = 5000

cmd_args.add('mnist:dataset_folder', 'str', '../MNIST_data/')


def maybe_download(filename, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    if not os.path.exists(work_directory):
        os.mkdir(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        filepath, _ = urllib.urlretrieve(SOURCE_URL + filename, filepath)
        statinfo = os.stat(filepath)
        log = logger.get()
        log.info('Succesfully downloaded {} {} bytes.'.format(
            filename, statinfo.st_size))
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    log = logger.get()
    log.info('Extracting {}'.format(filename))
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(filename, one_hot=False):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    log = logger.get()
    log.info('Extracting {}'.format(filename))
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return dense_to_one_hot(labels)
        return labels


class MNISTDataProvider(data_provider.DataProvider):

    def __init__(self, split='train', filename=None):
        super(MNISTDataProvider, self).__init__()
        self.log = logger.get()
        if split is None:
            self.split = 'train'
        else:
            self.split = split
        self.log.info('Data split: {}'.format(self.split))
        self.filename = filename
        self._images = None
        self._labels = None
        self.register_option('mnist:dataset_folder')
        pass

    def init_default_options(self):
        self.set_default_option('one_hot', True)
        pass

    def init_from_main(self):
        return super(MNISTDataProvider, self).init_from_main()

    def init_data(self):
        self.init_default_options()
        self.folder = self.get_option('mnist:dataset_folder')
        if self.filename is None:
            if self.split == 'train' or self.split == 'valid':
                self.image_filename = 'train-images-idx3-ubyte.gz'
                self.label_filename = 'train-labels-idx1-ubyte.gz'
            elif self.split == 'test':
                self.image_filename = 't10k-images-idx3-ubyte.gz'
                self.label_filename = 't10k-labels-idx1-ubyte.gz'
            else:
                raise Exception('Unknown split "{}"'.format(self.split))
        else:
            self.image_filename = filename + '-images-idx3-ubyte.gz'
            self.label_filename = filename + '-labels-idx1-ubyte.gz'

        local_file = maybe_download(self.image_filename, self.folder)
        self._images = extract_images(local_file)
        local_file = maybe_download(self.label_filename, self.folder)
        self._labels = extract_labels(
            local_file, one_hot=self.get_option('one_hot'))
        pass

    def get_size(self):
        if self._images is None:
            self.init_data()
        if self.split == 'valid':
            size = VALIDATION_SIZE
        elif self.split == 'train':
            size = self._images.shape[0] - VALIDATION_SIZE
        else:
            size = self._images.shape[0]
        self.log.info('Dataset size: {}'.format(size))
        return size

    def get_batch_idx(self, idx, **kwargs):
        """Return the next `batch_size` examples from this data set."""
        if self._images is None:
            self.init_data()
        if self.split == 'train':
            return {
                'x': self._images[VALIDATION_SIZE + numpy.array(idx)],
                'y_gt': self._labels[VALIDATION_SIZE + numpy.array(idx)]
            }
        else:
            return {
                'x': self._images[idx],
                'y_gt': self._labels[idx]
            }
        pass
    pass

data_provider.get_factory().register('mnist', MNISTDataProvider)
