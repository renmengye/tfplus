# import sys
# sys.path.insert(0, '/pkgs/tensorflow-gpu-0.9.0')
import tfplus
import os
import numpy as np
import synset
import cv2
import time


class ImageNetDataProvider(tfplus.data.DataProvider):

    def __init__(self, split='train', folder='/ais/gobi3/datasets/imagenet'):
        super(ImageNetDataProvider, self).__init__()
        self.log = tfplus.utils.logger.get()
        self._split = split
        self._folder = folder
        self._img_ids = None
        self._labels = None
        pass

    @property
    def folder(self):
        return self._folder

    @property
    def split(self):
        return self._split

    @property
    def img_ids(self):
        if self._img_ids is None:
            image_folder = os.path.join(self.folder, self.split)
            self._img_ids = []
            if self.split == 'train':
                self._labels = []
                folders = os.listdir(image_folder)
                for ff in folders:
                    subfolder = os.path.join(image_folder, ff)
                    image_fnames = os.listdir(subfolder)
                    self._img_ids.extend(image_fnames)
                    self._labels.extend(
                        [synset.get_index(ff)] * len(image_fnames))
            elif self.split == 'valid' or self.split == 'test':
                self._img_ids = os.listdir(image_folder)
                self._img_ids = sorted(self._img_ids)
                if self.split == 'valid':
                    self._labels = []
                    with open(os.path.join(
                            self.folder, 'synsets.txt'), 'r') as f_cls:
                        synsets = f_cls.readlines()
                    synsets = [ss.strip('\n') for ss in synsets]
                    with open(os.path.join(
                            self.folder, 'valid_labels.txt'), 'r') as f_lab:
                        labels = f_lab.readlines()
                    labels = [int(ll) for ll in labels]
                    slabels = [synsets[ll] for ll in labels]
                    self._labels = [synset.get_index(sl) for sl in slabels]
            self._labels = np.array(self._labels)
        return self._img_ids

    @property
    def labels(self):
        if self._labels is None and self.split != 'test':
            a = self.img_ids
        return self._labels

    def get_size(self):
        return len(self.img_ids)

    def get_batch_idx(self, idx, **kwargs):
        start_time = time.time()
        x = np.zeros([len(idx), 256, 256, 3], dtype='float32')
        y_gt = np.zeros([len(idx), 1000], dtype='float32')
        for kk, ii in enumerate(idx):
            if self.split == 'train':
                folder = os.path.join('train', self.img_ids[ii].split('_')[0])
            else:
                folder = self.split
            img_fname = os.path.join(self.folder, folder, self.img_ids[ii])
            # print img_fname
            x_ = cv2.imread(img_fname)
            x_ = cv2.resize(x_, (256, 256), interpolation=cv2.INTER_CUBIC)
            # print x_.mean()
            x[kk] = x_.astype('float32')
            y_gt[kk, self.labels[ii]] = 1.0
        # print 'Mean', x.mean()
        # x = x - self._img_mean
        # print 'Mean - mean', x.mean()
        results = {
            'x': x,
            'y_gt': y_gt
        }
        # self.log.info('Fetch data time: {:.4f} ms'.format(
        #               (time.time() - start_time) * 1000))
        return results
    pass


tfplus.data.data_provider.get_factory().register('imagenet',
                                                 ImageNetDataProvider)

if __name__ == '__main__':
    # print len(ImageNetDataProvider().img_ids)
    labels = ImageNetDataProvider(split='train').labels
    print labels.max()
    print labels.min()
    print len(labels)
    labels = ImageNetDataProvider(split='valid').labels
    print labels.max()
    print labels.min()
    print len(labels)
    img_ids = ImageNetDataProvider(split='valid').img_ids
    print img_ids[0]
    print img_ids[-1]
