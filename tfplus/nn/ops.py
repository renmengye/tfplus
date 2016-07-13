from graph_builder import GraphBuilder

import tensorflow as tf


def conv2d(x, w, stride=1):
    """2-D convolution.

    Args:
        x: input tensor, [B, H, W, D]
        w: filter tensor, [F, F, In, Out]
    """
    return tf.nn.conv2d(x, w, strides=[1, stride, stride, 1], padding='SAME')


class Conv2D(GraphBuilder):
    """2D convolution."""

    def __init__(self, w, stride=1):
        """
        Args:
            w: filter tensor, [F, F, In, Out]
            stride: int
        """
        super(Conv2D, self).__init__()
        self.w = w
        self.stride = stride
        pass

    def build(self, inp):
        x = self.get_single_input(inp)
        return tf.nn.conv2d(x, self.w,
                            strides=[1, self.stride, self.stride, 1],
                            padding='SAME')


class Conv2DW(GraphBuilder):

    def __init__(self, f, ch_in, ch_out, stride=1, wd=None, scope='conv'):
        super(Conv2DW, self).__init__()
        self.stride = stride
        self.f = f
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.scope = scope
        self.wd = wd
        pass

    def init_var(self):
        self.w = self.declare_var(
            [self.f, self.f, self.ch_in, self.ch_out], name='w', wd=self.wd)
        pass

    def build(self, inp):
        with tf.variable_scope(self.scope):
            self.lazy_init_var()
            x = self.get_single_input(inp)
            return tf.nn.conv2d(x, self.w,
                                strides=[1, self.stride, self.stride, 1],
                                padding='SAME')

    def get_save_var_dict(self):
        return {
            'w': self.w
        }


class Linear(GraphBuilder):

    def __init__(self, d_in, d_out, wd=None, scope='linear'):
        super(Linear, self).__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.wd = wd
        self.scope = scope
        pass

    def init_var(self):
        self.w = self.declare_var(
            [self.d_in, self.d_out], name='w', wd=self.wd)
        pass

    def build(self, inp):
        with tf.variable_scope(self.scope):
            self.lazy_init_var()
            x = self.get_single_input(inp)
            return tf.matmul(x, self.w)

    def get_save_var_dict(self):
        return {
            'w': self.w
        }


class DilatedConv2D(GraphBuilder):
    """Dilated 2D convolution."""

    def __init__(self, w, rate=1):
        super(DilatedConv2D, self).__init__()
        self.w = w
        self.rate = rate
        pass

    def build(self, inp):
        x = self.get_single_input(inp)
        return tf.nn.atrous_conv2d(x, self.w, rate=self.rate, padding='SAME')


def max_pool(x, ratio):
    """N x N max pooling.

    Args:
        x: input tensor, [B, H, W, D]
        ratio: N by N pooling ratio
    """
    return tf.nn.max_pool(x, ksize=[1, ratio, ratio, 1],
                          strides=[1, ratio, ratio, 1], padding='SAME')


class MaxPool(GraphBuilder):

    def __init__(self, kernel, stride=None):
        super(MaxPool, self).__init__()
        self.kernel = kernel
        if stride is None:
            self.stride = kernel
        else:
            self.stride = stride
        pass

    def build(self, inp):
        x = self.get_single_input(inp)
        return tf.nn.max_pool(x, ksize=[1, self.kernel, self.kernel, 1],
                              strides=[1, self.stride, self.stride, 1],
                              padding='SAME')


def avg_pool(x, ratio):
    """N x N max pooling.

    Args:
        x: input tensor, [B, H, W, D]
        ratio: N by N pooling ratio
    """
    return tf.nn.avg_pool(x, ksize=[1, ratio, ratio, 1],
                          strides=[1, ratio, ratio, 1], padding='SAME')


class AvgPool(GraphBuilder):

    def __init__(self, ratio):
        super(AvgPool, self).__init__()
        self.ratio = ratio
        pass

    def build(self, inp):
        x = self.get_single_input(inp)
        return tf.nn.avg_pool(x, ksize=[1, self.ratio, self.ratio, 1],
                              strides=[1, self.ratio, self.ratio, 1],
                              padding='SAME')


def bce(y_out, y_gt):
    eps = 1e-5
    ce = -y_gt * tf.log(y_out + eps) - (1 - y_gt) * tf.log(1 - y_out + eps)
    return ce


class BCE(GraphBuilder):

    def __init__(self):
        super(BCE, self).__init__()
        pass

    def build(self, inp):
        y_out = inp['y_out']
        y_gt = inp['y_gt']
        eps = 1e-5
        ce = -y_gt * tf.log(y_out + eps) - (1 - y_gt) * tf.log(1 - y_out + eps)
        return ce


class CE(GraphBuilder):

    def __init__(self):
        super(CE, self).__init__()
        pass

    def build(self, inp):
        y_out = inp['y_out']
        y_gt = inp['y_gt']
        eps = 1e-5
        ce = -y_gt * tf.log(y_out + eps)
        return ce


def dropout(x, keep_prob, phase_train):
    """Add dropout layer"""
    phase_train_f = tf.to_float(phase_train)
    keep_prob = (1.0 - phase_train_f) * 1.0 + phase_train_f * keep_prob
    return tf.nn.dropout(x, keep_prob)


class Dropout(GraphBuilder):

    def __init__(self, keep_prob, phase_train):
        super(Dropout, self).__init__()
        self.keep_prob = keep_prob
        self.phase_train = phase_train
        pass

    def build(self, inp):
        phase_train_f = tf.to_float(phase_train)
        keep_prob = (1.0 - phase_train_f) * 1.0 + phase_train_f * keep_prob
        return tf.nn.dropout(x, keep_prob)
