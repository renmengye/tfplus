from __future__ import division

import cv2
import numpy as np
import tensorflow as tf


class ImagePreprocessor(object):

    def __init__(self, resize=256, rnd_hflip=True, rnd_resize=[256, 480],
                 crop=224, rnd_colour=True, resize_base='short'):
        """
        """
        self._random = np.random.RandomState(2)
        # readonly
        self._rnd_resize = rnd_resize
        # readonly
        self._rnd_hflip = rnd_hflip
        # readonly
        self._resize = resize
        # readonly
        self._crop = crop
        # readonly
        self._rnd_colour = rnd_colour
        # readonly
        self._resize_base = resize_base
        self._image_in, self._image_out = self.build_colour_graph()
        self._sess = tf.Session()
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

    def get_resize(old_size, siz):
        width = old_size[0]
        height = old_size[1]
        if self.resize_base == 'short':
            if width < height:
                siz2 = (siz, int(height / width * siz))
            else:
                siz2 = (int(width / height * siz), siz)
            pad = [0.0, 0.0]
        elif self.resize_base == 'long':
            # For now, just centre padding...
            if width < height:
                siz2 = (int(width / height * siz), siz)
                pad = [siz - siz2[0], 0.0]
            else:
                siz2 = (siz, int(height / width * siz))
                pad = [0.0, siz - siz2[1]]
        elif self.resize_base == 'squeeze':
            siz2 = (siz, siz)
            pad = [0.0, 0.0]
        else:
            raise Exception('Unknown resize base {}'.format(resize_base))
        return siz2, pad

    def redraw(self, old_size):
        siz = int(self.random.uniform(self.rnd_resize[0], self.rnd_resize[1]))
        siz2 = self.get_resize(old_size, siz)
        siz3 = [siz2[0] + pad[0], siz2[1] + pad[1]]
        offset = [0.0, 0.0]
        offset[0] = int(self.random.uniform(0.0, siz3[0] - self.crop))
        offset[1] = int(self.random.uniform(0.0, siz3[1] - self.crop))
        hflip = bool(self.random.uniform(0, 1))
        return {
            'offset': offset,
            'pad': pad,
            'resize': siz2,
            'hflip': hflip
        }

    def process(self, image, rnd=True, resize_base='short', rnd_package=None):
        """Process the images.

        redraw: Whether to redraw random numbers used for random cropping, and
        horizontal flipping. Random colours have to be redrawn.
        """
        # BGR => RGB
        image = image[:, :, [2, 1, 0]]
        # [0, 255] => [0, 1]
        image = (image / 255).astype('float32')
        width = image.shape[1]
        height = image.shape[0]

        if rnd_package is None:
            rnd_package = self.redraw((width, height))

        offset_ = rnd_package['offset']
        resize_ = rnd_package['resize']
        hflip_ = rnd_package['hflip']

        if rnd:
            # Random resize, random crop
            resize = resize_
            offset = offset_
            hflip = hflip_ and self.rnd_hflip
        else:
            # Fixed resize, center crop
            siz = self.resize
            short = min(width, height)
            lon = max(width, height)
            resize, pad = get_resize((width, height), self.resize,
                                     resize_base=resize_base)
            offset = [0, 0]
            if self.resize_base == 'short':
                offset[0] = int((resize[0] - self.crop) / 2)
                offset[1] = int((resize[1] - self.crop) / 2)
            hflip = False

        image = cv2.resize(image, resize, interpolation=cv2.INTER_CUBIC)
        
        if pad[0] > 0 or pad[1] > 0:
            image = np.pad(image, [[pad[1], pad[1]], [pad[0], pad[0]], [0, 0]])

        if image.shape[0] != self.crop or image.shape[1] != self.crop:
            image = image[offset[1]: self.crop + offset[1],
                          offset[0]: self.crop + offset[0], :]

        if hflip:
            image = np.fliplr(image)

        if rnd and self.rnd_colour and image.shape[-1] == 3:
            image = self.sess.run(self.image_out, feed_dict={
                                  self.image_in: image})
        # RGB => BGR
        image = image[:, :, [2, 1, 0]]

        return image, rnd_package
