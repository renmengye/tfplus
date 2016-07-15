from __future__ import division

import cv2
import numpy as np
import tensorflow as tf


class ImagePreprocessor(object):

    def __init__(self, resize, rnd_hflip, rnd_resize, crop, rnd_colour):
        """
        """
        self._random = np.random.RandomState(2)
        self._rnd_resize = rnd_resize
        self._rnd_hflip = rnd_hflip
        self._resize = resize
        self._crop = crop
        self._rnd_colour = rnd_colour
        self._image_in, self._image_out = self.build_colour_graph()
        self.redraw()
        self._sess = tf.Session()
        pass

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
            image_out = tf.image.random_hue(image_out, max_delta=0.2)
            image_out = tf.image.random_contrast(
                image_out, lower=0.5, upper=1.5)
            image_out = tf.clip_by_value(image_out, 0.0, 1.0)
        return image_in, image_out

    def redraw(self, height, width):
        self.siz = int(self._random.uniform(
            self._rnd_resize[0], self._rnd_resize[1]))
        short = min(width, height)
        lon = max(width, height)
        if width < height:
            ratio = self.siz / width
            self.siz2 = (self.siz, height * ratio)
        else:
            ratio = self.siz / height
            self.siz2 = (width * ratio, self.siz)
        self.offset = [0.0, 0.0]
        self.offset[0] = int(self._random.uniform(
            0.0, self.siz2[0] - self._crop))
        self.offset[1] = int(self._random.uniform(
            0.0, self.siz2[1] - self._crop))
        self.hflip = bool(self._random.uniform(0, 1))
        pass

    def process(self, image, rnd=True, redraw=True):
        """Process the images.

        redraw: Whether to redraw random numbers used for random cropping, and
        horizontal flipping. Random colours have to be redrawn.
        """
        if redraw:
            self.redraw(height=image.shape[0], width=image.shape[1])

        if rnd:
            siz2 = self.siz2
            offset = self.offset
            hflip = self.hflip and self._rnd_hflip
        else:
            siz2 = (self._resize, self._resize)
            offset = (self._rnd_resize[0] - self._crop) / 2
            offset = [offset, offset]
            hflip = False

        image = cv2.resize(image, siz2, interpolation=cv2.INTER_CUBIC)
        image = image[offset[0]: self._crop + offset[0],
                      offset[1]: self._crop + offset[1], :]
        if hflip:
            image = np.fliplr(image)

        if rnd and self._rnd_colour and image.shape[-1] == 3:
            image = self._sess.run(self._image_out, feed_dict={
                self._image_in: image})

        return image
