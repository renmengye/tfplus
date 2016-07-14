import cslab_environ

import tfplus
import tensorflow as tf

from tfplus.nn import ResNet, MaxPool, AvgPool, BatchNorm, Linear, Conv2DW


class ResNetImageNetModel(tfplus.nn.ContainerModel):

    def __init__(self, nlayers=None):
        self.nlayers = nlayers
        self._img_mean = np.array(
            [103.062623801, 115.902882574, 123.151630838], dtype='float32')
        super(ResNetImageNetModel, self).__init__()
        pass

    def init_default_options(self):
        """Default option is a 50-layer ResNet"""
        self.set_default_option('inp_height', 224)
        self.set_default_option('inp_width', 224)
        self.set_default_option('inp_depth', 3)
        self.set_default_option('num_classes', 1000)
        self.set_default_option('layers', [3, 4, 6, 3])
        self.set_default_option('channels', [256, 256, 512, 1024, 2048])
        self.set_default_option('bottleneck', True)
        self.set_default_option('shortcut', 'projection')
        self.set_default_option('strides', [1, 2, 2, 2, 2])
        self.set_default_option('weight_decay', 0.00004)
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
        self.conv1 = Conv2DW(
            f=7, ch_in=inp_depth, ch_out=channels[0], stride=2, wd=wd,
            scope='conv')
        self.res_net = ResNet(layers=layers,
                              channels=channels,
                              strides=strides,
                              bottleneck=bottleneck,
                              shortcut=shortcut,
                              wd=wd)
        self.fc = Linear(d_in=channels[-1],
                         d_out=num_classes, wd=wd, scope='fc')
        pass

    def build_input(self):
        inp_height = self.get_option('inp_height')
        inp_width = self.get_option('inp_width')
        inp_depth = self.get_option('inp_depth')
        num_classes = self.get_option('num_classes')
        x = self.add_input_var(
            'x', [None, inp_height, inp_width, inp_depth], 'float')
        phase_train = self.add_input_var('phase_train', None, 'bool')
        y_gt = self.add_input_var(
            'y_gt', [None, num_classes], 'float')
        return {
            'x': x,
            'phase_train': phase_train,
            'y_gt': y_gt
        }

    def build(self, inp):
        channels = self.get_option('channels')
        self.lazy_init_var()
        x = inp['x']
        phase_train = inp['phase_train']
        x = x - self._img_mean
        # x = tf.Print(x, [tf.reduce_mean(x)])
        h = self.conv1(x)
        # h = tf.Print(h, [tf.reduce_mean(h), tf.reduce_sum(h),
        #                    tf.reduce_mean(self.conv1.w)])
        self.bn1 = BatchNorm(h.get_shape()[-1])
        h = self.bn1({'input': h, 'phase_train': phase_train})
        h = tf.nn.relu(h)
        h = MaxPool(3, stride=2)(h)
        self.log.info('Before ResNet shape: {}'.format(h.get_shape()))
        # h = tf.Print(h, [tf.reduce_mean(h), 0.0])
        h = self.res_net({'input': h, 'phase_train': phase_train})
        # h = tf.Print(h, [tf.reduce_mean(h), 1.0])
        self.log.info('Before AvgPool shape: {}'.format(h.get_shape()))
        h = AvgPool(7)(h)
        # h = tf.Print(h, [tf.reduce_mean(h), 2.0])
        h = tf.reshape(h, [-1, channels[-1]])
        self.log.info('Before FC shape: {}'.format(h.get_shape()))
        y_out = self.fc(h)
        y_out = tf.nn.softmax(y_out)
        # y_out = tf.Print(y_out, [tf.reduce_mean(y_out), 3.0])
        return y_out

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        self.add_prefix_to('conv1', self.conv.get_save_var_dict(), results)
        self.add_prefix_to('bn1', self.bn1.get_save_var_dict(), results)
        self.add_prefix_to(
            'res_net', self.res_net.get_save_var_dict(), results)
        self.add_prefix_to('fc', self.fc.get_save_var_dict(), results)
        self.log.info('Save variable list:')
        [self.log.info((v[0], v[1].name)) for v in results.items()]
        return results

    pass

if __name__ == '__main__':
    m = ResNetImageNetModel()
    m.build(m.build_input())
