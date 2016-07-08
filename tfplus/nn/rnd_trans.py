from graph_builder import GraphBuilder

from tensorflow.python.framework import ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
import tensorflow as tf


class ImageRandomTransform(GraphBuilder):

    def __init__(self, padding, rnd_vflip=True, rnd_hflip=True,
                 rnd_transpose=True, rnd_size=False, _debug=False):
        super(ImageRandomTransform, self).__init__()
        self.padding = padding
        self.rnd_vflip = rnd_vflip
        self.rnd_hflip = rnd_hflip
        self.rnd_transpose = rnd_transpose
        self.rnd_size = rnd_size
        self._debug = _debug
        pass

    def init_var(self):
        self.rand_h = tf.random_uniform([1], 1.0 - float(self.rnd_hflip), 1.0)
        self.rand_v = tf.random_uniform([1], 1.0 - float(self.rnd_vflip), 1.0)
        self.rand_t = tf.random_uniform(
            [1], 1.0 - float(self.rnd_transpose), 1.0)
        self.offset = tf.random_uniform(
            [2], dtype='int32', maxval=self.padding * 2)
        if self._debug:
            self.offset = tf.Print(self.offset, ['Forward RND module', self.offset])
        if self.rnd_size:
            self.space = 2 * self.padding - self.offset
            self.offset20 = tf.random_uniform(
                [], dtype='int32', maxval=self.space[0] * 2) - self.space[0]
            self.offset21 = tf.random_uniform(
                [], dtype='int32', maxval=self.space[1] * 2) - self.space[1]
            self.offset2 = tf.pack([self.offset20, self.offset21])
        else:
            self.offset2 = tf.zeros([2], dtype='int32')
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['input']
        phase_train = inp['phase_train']
        if 'rnd_colour' in inp:
            rnd_colour = inp['rnd_colour']
        else:
            rnd_colour = False
        if 'axis' in inp:
            axis = inp['axis']
        else:
            axis = 3

        padding = self.padding
        offset = self.offset
        offset2 = self.offset2
        rand_h = self.rand_h
        rand_v = self.rand_v
        rand_t = self.rand_t

        xshape = tf.shape(x)
        phase_train_f = tf.to_float(phase_train)

        if axis == 3:
            inp_height = xshape[1]
            inp_width = xshape[2]
            # Add padding
            x_pad = tf.pad(
                x, [[0, 0], [padding, padding], [padding, padding], [0, 0]])
            # Random crop
            x_rand = tf.slice(x_pad, tf.pack([0, offset[0], offset[1], 0]),
                              tf.pack([-1, inp_height + offset2[0],
                                       inp_width + offset2[1], -1]))
            if self.rnd_size:
                x_rand = tf.image.resize_nearest_neighbor(
                    x_rand, tf.pack([inp_height, inp_width]))
            x_ctr = tf.slice(x_pad, [0, padding, padding, 0],
                             tf.pack([-1, inp_height, inp_width, -1]))
            mirror = tf.pack([1.0, rand_v[0], rand_h[0], 1.0]) < 0.5
            x_rand = tf.reverse(x_rand, mirror)
            do_tr = tf.cast(rand_t[0] < 0.5, 'int32')
            x_rand = tf.transpose(x_rand, tf.pack(
                [0, 1 + do_tr, 2 - do_tr, 3]))
        elif axis == 1:
            inp_height = xshape[2]
            inp_width = xshape[3]
            x_pad = tf.pad(
                x, [[0, 0], [0, 0], [padding, padding], [padding, padding]])
            x_rand = tf.slice(x_pad, tf.pack([0, 0, offset[0], offset[1]]),
                              tf.pack([-1, -1, inp_height + offset2[0],
                                       inp_width + offset2[1]]))
            # if self.rnd_size:
            #     x_rand = tf.transpose(x_rand, [0, 2, 3, 1])
            #     x_rand = tf.image.resize_nearest_neighbor(
            #         x_rand, tf.pack([inp_height, inp_width]))
            #     x_rand = tf.transpose(x_rand, [0, 3, 1, 2])

            x_ctr = tf.slice(x_pad, [0, 0, padding, padding],
                             tf.pack([-1, -1, inp_height, inp_width]))
            mirror = tf.pack([1.0, 1.0, rand_v[0], rand_h[0]]) < 0.5
            x_rand = tf.reverse(x_rand, mirror)
            do_tr = tf.cast(rand_t[0] < 0.5, 'int32')
            x_rand = tf.transpose(x_rand, tf.pack(
                [0, 1, 2 + do_tr, 3 - do_tr]))

        if rnd_colour:
            x_rand = random_hue(x_rand, 0.05)
            x_rand = random_saturation(x_rand, 0.95, 1.05)
            x_rand = tf.image.random_brightness(x_rand, 0.05)
            x_rand = tf.image.random_contrast(x_rand, 0.95, 1.05)

        x = (1.0 - phase_train_f) * x_ctr + phase_train_f * x_rand
        return {'trans': x}
    pass


def adjust_hue(image, delta, name=None):
    with ops.op_scope([image], name, 'adjust_hue') as name:
        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        flt_image = tf.image.convert_image_dtype(image, tf.float32)

        hsv = gen_image_ops.rgb_to_hsv(flt_image)

        hue = tf.slice(hsv, [0, 0, 0, 0], [-1, -1, -1, 1])
        saturation = tf.slice(hsv, [0, 0, 0, 1], [-1, -1, -1, 1])
        value = tf.slice(hsv, [0, 0, 0, 2], [-1, -1, -1, 1])

        # Note that we add 2*pi to guarantee that the resulting hue is a positive
        # floating point number since delta is [-0.5, 0.5].
        hue = math_ops.mod(hue + (delta + 1.), 1.)

        hsv_altered = tf.concat(3, [hue, saturation, value])
        rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

        return tf.image.convert_image_dtype(rgb_altered, orig_dtype)


def adjust_saturation(image, saturation_factor, name=None):
    with ops.op_scope([image], name, 'adjust_saturation') as name:
        # Remember original dtype to so we can convert back if needed
        orig_dtype = image.dtype
        flt_image = tf.image.convert_image_dtype(image, tf.float32)

        hsv = gen_image_ops.rgb_to_hsv(flt_image)

        hue = tf.slice(hsv, [0, 0, 0, 0], [-1, -1, -1, 1])
        saturation = tf.slice(hsv, [0, 0, 0, 1], [-1, -1, -1, 1])
        value = tf.slice(hsv, [0, 0, 0, 2], [-1, -1, -1, 1])

        saturation *= saturation_factor
        saturation = clip_ops.clip_by_value(saturation, 0.0, 1.0)

        hsv_altered = tf.concat(3, [hue, saturation, value])
        rgb_altered = gen_image_ops.hsv_to_rgb(hsv_altered)

        return tf.image.convert_image_dtype(rgb_altered, orig_dtype)


def random_flip_left_right(image, seed=None):
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror = math_ops.less(tf.pack(
        [1.0, 1.0, uniform_random, 1.0]), 0.5)
    return tf.reverse(image, mirror)


def random_flip_up_down(image, seed=None):
    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=seed)
    mirror = math_ops.less(tf.pack(
        [1.0, uniform_random, 1.0, 1.0]), 0.5)
    return tf.reverse(image, mirror)


def random_hue(image, max_delta, seed=None):
    if max_delta > 0.5:
        raise ValueError('max_delta must be <= 0.5.')

    if max_delta < 0:
        raise ValueError('max_delta must be non-negative.')

    delta = random_ops.random_uniform([], -max_delta, max_delta, seed=seed)
    return adjust_hue(image, delta)


def random_saturation(image, lower, upper, seed=None):
    if upper <= lower:
        raise ValueError('upper must be > lower.')

    if lower < 0:
        raise ValueError('lower must be non-negative.')

    # Pick a float in [lower, upper]
    saturation_factor = random_ops.random_uniform([], lower, upper, seed=seed)
    return adjust_saturation(image, saturation_factor)
