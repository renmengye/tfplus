from __future__ import division

import cv2
import numpy as np
import tensorflow as tf


class ImagePreprocessor(object):

    def __init__(self, resize=[256, 256], rnd_hflip=True,
                 rnd_resize=[256, 256, 480, 480],
                 crop=[224, 224], rnd_colour=True, resize_base='short'):
        """
        """
        self._random = np.random.RandomState(2)
        # readonly
        # [w1, h1, w2, h2] (lower and upper bound of random resize)
        if len(rnd_resize) == 2:
            rnd_resize = [rnd_resize[0], rnd_resize[0],
                          rnd_resize[1], rnd_resize[1]]
        self._rnd_resize = rnd_resize

        # readonly
        self._rnd_hflip = rnd_hflip
        # readonly
        if type(resize) == int:
            resize = [resize, resize]
        self._resize = resize
        # readonly
        if type(crop) == int:
            crop = [crop, crop]
        self._crop = crop
        # readonly
        self._rnd_colour = rnd_colour
        # readonly
        self._resize_base = resize_base
        self._image_in, self._image_out = self.build_colour_graph()
        if rnd_colour:
            self._sess = tf.Session()
        else:
            self._sess = None
        pass

    @property
    def crop(self):
        return self._crop

    @property
    def resize(self):
        return self._resize

    @property
    def rnd_hflip(self):
        return self._rnd_hflip

    @property
    def rnd_resize(self):
        return self._rnd_resize

    @property
    def random(self):
        return self._random

    @property
    def rnd_colour(self):
        return self._rnd_colour

    @property
    def resize_base(self):
        return self._resize_base

    @property
    def image_in(self):
        return self._image_in

    @property
    def image_out(self):
        return self._image_out

    @property
    def sess(self):
        return self._sess

    def build_colour_graph(self):
        """Build random colour graph."""
        device = '/cpu:0'
        with tf.device(device):
            image_in = tf.placeholder('float', [None, None, 3])
            image_out = image_in
            image_out = tf.image.random_brightness(
                image_out, max_delta=32. / 255.)
            image_out = tf.image.random_saturation(
                image_out, lower=0.5, upper=1.5)
            image_out = tf.image.random_hue(image_out, max_delta=0.1)
            image_out = tf.image.random_contrast(
                image_out, lower=0.5, upper=1.5)
            image_out = tf.clip_by_value(image_out, 0.0, 1.0)
        return image_in, image_out

    def get_resize(self, old_size, siz):
        """
        old_siz: [W, H]
        siz: [W', H']

        short: Resize base on the shorter axis, and crop.
        long: Resize base on the longer axis, and pad.
        pad: Just pad to the right size.
        squeeze: Just squeeze to the right size.
        """
        w = old_size[0]
        h = old_size[1]
        new_w = siz[0]
        new_h = siz[1]
        if self.resize_base == 'short':
            if w < h:
                ratio = [new_w / w, new_w / w]
                siz2 = (new_w, int(h / w * new_w))
            else:
                ratio = [new_h / h, new_h / h]
                siz2 = (int(w / h * new_h), new_h)
            pad = [0, 0]
        elif self.resize_base == 'long':
            # For now, just centre padding...
            if width < height:
                siz2 = (int(w / h * new_h / 2) * 2, new_h)
                ratio = [new_h / h, new_h / h]
                pad = [int((new_w - siz2[0]) / 2), 0]
            else:
                siz2 = (new_w, int(h / w * new_w / 2) * 2)
                ratio = [new_w / w, new_w / w]
                pad = [0, int((new_h - siz2[1]) / 2)]
        elif self.resize_base == 'pad':
            siz2 = old_size
            ratio = [1.0, 1.0]
            pad = [int((new_w - w) / 2), int((new_h - h) / 2)]
        elif self.resize_base == 'squeeze':
            siz2 = siz
            pad = [0, 0]
            ratio = [new_w / w, new_h / h]
        else:
            raise Exception('Unknown resize base {}'.format(self.resize_base))
        return siz2, pad, ratio

    def redraw(self, old_size, rnd=True):
        if rnd:
            siz_w = int(self.random.uniform(
                self.rnd_resize[0], self.rnd_resize[2]))
            siz_h = int(self.random.uniform(
                self.rnd_resize[1], self.rnd_resize[3]))
            siz = (siz_w, siz_h)
            siz2, pad, ratio = self.get_resize(old_size, siz)
            siz3 = [siz2[0] + pad[0], siz2[1] + pad[1]]
            offset = [0.0, 0.0]
            offset[0] = int(self.random.uniform(0.0, siz3[0] - self.crop[0]))
            offset[1] = int(self.random.uniform(0.0, siz3[1] - self.crop[1]))
            hflip = self.random.uniform(0, 1) > 0.5
            return {
                'offset': offset,
                'pad': pad,
                'ratio': ratio,
                'resize': siz2,
                'hflip': hflip and self.rnd_hflip
            }
        else:
            width = old_size[0]
            height = old_size[1]
            siz = self.resize
            short = min(width, height)
            lon = max(width, height)
            resize, pad, ratio = self.get_resize((width, height), self.resize)
            offset = [0, 0]
            if self.resize_base == 'short':
                offset[0] = int((resize[0] - self.crop[0]) / 2)
                offset[1] = int((resize[1] - self.crop[1]) / 2)
            hflip = False
            return {
                'offset': offset,
                'pad': pad,
                'ratio': ratio,
                'resize': resize,
                'hflip': hflip
            }

    def process(self, image, rnd=True, rnd_package=None, div_255=True):
        """Process the images.

        redraw: Whether to redraw random numbers used for random cropping, and
        horizontal flipping. Random colours have to be redrawn.
        """
        # BGR => RGB
        if rnd and self.rnd_colour and \
                len(image.shape) == 3 and image.shape[-1] == 3:
            image = image[:, :, [2, 1, 0]]
        # [0, 255] => [0, 1]
        if div_255:
            image = (image / 255.0).astype('float32')
        width = image.shape[1]
        height = image.shape[0]

        if rnd_package is None:
            rnd_package = self.redraw((width, height), rnd=rnd)
        resize = rnd_package['resize']
        pad = rnd_package['pad']
        hflip = rnd_package['hflip']
        offset = rnd_package['offset']

        image = cv2.resize(image, (resize[0], resize[1]),
                           interpolation=cv2.INTER_CUBIC)

        if pad[0] > 0 or pad[1] > 0:
            image = np.pad(image, [[pad[1], pad[1]], [pad[0], pad[0]], [0, 0]],
                           'constant', constant_values=(0,))

        if image.shape[0] != self.crop[0] or image.shape[1] != self.crop[1]:
            image = image[offset[1]: self.crop[1] + offset[1],
                          offset[0]: self.crop[0] + offset[0]]

        if hflip:
            image = np.fliplr(image)

        if rnd and self.rnd_colour and \
                len(image.shape) == 3 and image.shape[-1] == 3:
            image = self.sess.run(self.image_out, feed_dict={
                self.image_in: image})
            # RGB => BGR
            image = image[:, :, [2, 1, 0]]
        image = np.minimum(np.maximum(image, 0.0), 1.0)

        return image, rnd_package
