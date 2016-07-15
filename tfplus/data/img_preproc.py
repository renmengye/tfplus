import cv2
import tensorflow as tf


class ImagePreprocessor(object):

    def __init__(self, resize, rnd_hflip, rnd_resize, crop, rnd_colour):
        self._rnd_resize = rnd_resize
        self._rnd_hflip = rnd_hflip
        self._resize = resize
        self._crop = crop
        self._rnd_colour = rnd_colour
        self._image_in, self._image_out = self.build_colour_graph()
        self._sess = tf.Session()
        pass

    def build_colour_graph():
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

    def process(self, image, rnd=True, redraw=True):
        """Process the images.

        redraw: Whether to redraw random numbers used for random cropping, and
        horizontal flipping. Random colours have to be redrawn.
        """
        if rnd:
            siz = int(self._random.uniform(
                self._rnd_resize[0], self._rnd_resize[1]))
            offset = [0, 0]
            offset[0] = int(self._random.uniform(0.0, siz - self._crop))
            offset[1] = int(self._random.uniform(0.0, siz - self._crop))
            if self._rnd_hflip:
                hflip = bool(self._random.uniform(0, 1))
            else:
                hflip = False

        else:
            siz = self._resize
            offset = (self._rnd_resize[0] - self._crop) / 2
            offset = [offset, offset]
            pass

        image = cv2.resize(image, (siz, siz), interpolation=cv2.INTER_CUBIC)
        image = image[offset[0]: self._crop + offset[0],
                      offset[1]: self._crop + offset[1], :]
        if hflip:
            image = np.fliplr(image)

        if rnd and self._rnd_colour:
            image = self._sess.run(self._image_out, feed_dict={
                self._image_in: image})

        pass
