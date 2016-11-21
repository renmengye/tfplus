import cslab_environ

import numpy as np
import tfplus
import tensorflow as tf

from tfplus.nn import ResNetSBN, MaxPool, AvgPool, BatchNorm, Linear, Conv2DW, NewBatchNorm


class ResNetImageNetModel(tfplus.nn.Model):

    def __init__(self, nlayers=None, name='resnet_imagenet'):
        super(ResNetImageNetModel, self).__init__(name=name)
        self.nlayers = nlayers
        self._img_mean = np.array(
            [103.062623801, 115.902882574, 123.151630838], dtype='float')
        pass

    def init_default_options(self):
        """Default option is a 50-layer ResNet"""
        self.set_default_option('inp_depth', 3)
        self.set_default_option('num_classes', 1000)
        self.set_default_option('layers', [3, 4, 6, 3])
        self.set_default_option('channels', [64, 256, 512, 1024, 2048])
        self.set_default_option('bottleneck', True)
        self.set_default_option('shortcut', 'projection')
        self.set_default_option('strides', [1, 2, 2, 2])
        self.set_default_option('weight_decay', 0.00004)
        self.set_default_option('compatible', False)
        self.set_default_option('subtract_mean', False)
        self.set_default_option('trainable', True)
        self.set_default_option('new_bn', False)
        pass

    def init_var(self):
        channels = self.get_option('channels')
        inp_depth = self.get_option('inp_depth')
        wd = self.get_option('weight_decay')
        layers = self.get_option('layers')
        strides = self.get_option('strides')
        channels = self.get_option('channels')
        num_classes = self.get_option('num_classes')
        bottleneck = self.get_option('bottleneck')
        shortcut = self.get_option('shortcut')
        compatible = self.get_option('compatible')
        trainable = self.get_option('trainable')
        new_bn = self.get_option('new_bn')
        if new_bn:
            self.bn_cls = NewBatchNorm
            self.log.info('Using new batch norm')
        else:
            self.bn_cls = BatchNorm
        self.conv1 = Conv2DW(
            f=7, ch_in=inp_depth, ch_out=channels[0], stride=2, wd=wd,
            scope='conv', bias=False, trainable=trainable)
        self.bn1 = self.bn_cls(channels[0], trainable=trainable)
        self.res_net = ResNetSBN(layers=layers,
                                 channels=channels,
                                 strides=strides,
                                 bottleneck=bottleneck,
                                 shortcut=shortcut,
                                 compatible=compatible,
                                 trainable=trainable,
                                 wd=wd, new_bn=new_bn)
        self.fc = Linear(d_in=channels[-1],
                         d_out=num_classes, wd=wd, scope='fc', bias=True,
                         trainable=trainable)
        pass

    def build_input(self):
        inp_depth = self.get_option('inp_depth')
        num_classes = self.get_option('num_classes')
        x = self.add_input_var(
            'x', [None, None, None, inp_depth], 'float')
        phase_train = self.add_input_var('phase_train', None, 'bool')
        y_gt = self.add_input_var(
            'y_gt', [None, num_classes], 'float')
        return {
            'x': x,
            'phase_train': phase_train,
            'y_gt': y_gt
        }

    def build(self, inp):
        self.lazy_init_var()
        x = inp['x']
        phase_train = inp['phase_train']
        subtract_mean = self.get_option('subtract_mean')
        if subtract_mean:
            x = x * 255.0 - self._img_mean  # raw 0-255 values.
        # else:
        #     x = x * 2.0 - 1.0       # center at [-1, 1].
        
        if not self.has_var('x_sub'):
            self.register_var('x_sub', x)

        h = self.conv1(x)
        if not self.has_var('h_conv1'):
            self.register_var('h_conv1', h)
      
        h = self.bn1({'input': h, 'phase_train': phase_train})

        if not self.has_var('h_bn1'):
            self.register_var('h_bn1', h)
        h = tf.nn.relu(h)
        h = MaxPool(3, stride=2)(h)
        h = self.res_net({'input': h, 'phase_train': phase_train})
        
        if not self.has_var('h_last'):
            self.register_var('h_last', h)
        h = tf.reduce_mean(h, [1, 2])
        y_out = self.fc(h)
        y_out = tf.nn.softmax(y_out)
        return y_out

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        self.add_prefix_to('conv1', self.conv1.get_save_var_dict(), results)
        self.add_prefix_to('bn1', self.bn1.get_save_var_dict(), results)
        self.add_prefix_to(
            'res_net', self.res_net.get_save_var_dict(), results)
        self.add_prefix_to('fc', self.fc.get_save_var_dict(), results)
        return results

    pass

if __name__ == '__main__':
    m = ResNetImageNetModel()
    m.build(m.build_input())

    for v in tf.all_variables():
        print v.name, v.get_shape()

    for key in sorted(m.get_save_var_dict().keys()):
        print key
