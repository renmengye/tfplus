import tensorflow as tf
import tfplus

from resnet_imagenet_model import ResNetImageNetModel

NUM_CLS = 1000

tfplus.cmd_args.add('inp_depth', 'int', 3)
tfplus.cmd_args.add('layers', 'list<int>', [3, 4, 6, 3])
tfplus.cmd_args.add('strides', 'list<int>', [1, 2, 2, 2])
tfplus.cmd_args.add('channels', 'list<int>', [256, 256, 512, 1024, 2048])
tfplus.cmd_args.add('compatible', 'bool', False)
tfplus.cmd_args.add('subtract_mean', 'bool', False)
tfplus.cmd_args.add('shortcut', 'str', 'projection')
tfplus.cmd_args.add('learn_rate', 'float', 0.01)
tfplus.cmd_args.add('learn_rate_decay', 'float', 0.1)
tfplus.cmd_args.add('steps_per_lr_decay', 'int', 160000)
tfplus.cmd_args.add('momentum', 'float', 0.9)
tfplus.cmd_args.add('optimizer', 'str', 'momentum')
tfplus.cmd_args.add('wd', 'float', 1e-4)


class ResNetImageNetModelWrapper(tfplus.nn.ContainerModel):

    def __init__(self, name='resnet_imagenet_wrapper'):
        super(ResNetImageNetModelWrapper, self).__init__(name=name)
        self.register_option('inp_depth')
        self.register_option('layers')
        self.register_option('strides')
        self.register_option('channels')
        self.register_option('compatible')
        self.register_option('subtract_mean')
        self.register_option('bottleneck')
        self.register_option('shortcut')
        self.register_option('learn_rate')
        self.register_option('learn_rate_decay')
        self.register_option('steps_per_lr_decay')
        self.register_option('momentum')
        self.register_option('optimizer')
        self.register_option('wd')
        self.res_net = ResNetImageNetModel()
        self.add_sub_model(self.res_net)
        pass

    def init_default_options(self):
        self.set_default_option('bottleneck', True)
        self.set_default_option('optimizer', 'momentum')
        pass

    def build_input(self):
        inp_depth = self.get_option('inp_depth')
        x = self.add_input_var(
            'x', [None, None, None, inp_depth], 'float')
        y_gt = self.add_input_var('y_gt', [None, NUM_CLS], 'float')
        phase_train = self.add_input_var('phase_train', None, 'bool')
        return {
            'x': x,
            'y_gt': y_gt,
            'phase_train': phase_train
        }

    def init_var(self):
        inp_depth = self.get_option('inp_depth')
        layers = self.get_option('layers')
        strides = self.get_option('strides')
        channels = self.get_option('channels')
        bottleneck = self.get_option('bottleneck')
        compatible = self.get_option('compatible')
        subtract_mean = self.get_option('subtract_mean')
        shortcut = self.get_option('shortcut')
        wd = self.get_option('wd')
        phase_train = self.get_input_var('phase_train')
        self.res_net.set_all_options({
            'inp_depth': inp_depth,
            'layers': layers,
            'strides': strides,
            'channels': channels,
            'bottleneck': bottleneck,
            'subtract_mean': subtract_mean,
            'shortcut': shortcut,
            'compatible': compatible,
            'weight_decay': wd
        })
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['x']
        phase_train = inp['phase_train']
        x = tf.identity(x)
        self.register_var('x_trans', x)
        y_out = self.res_net({'x': x, 'phase_train': phase_train})
        self.register_var('y_out', y_out)
        return {
            'y_out': y_out
        }

    def build_loss(self, inp, output):
        y_gt = inp['y_gt']
        y_out = output['y_out']
        ce = tfplus.nn.CE()({'y_gt': y_gt, 'y_out': y_out})
        num_ex_f = tf.to_float(tf.shape(inp['x'])[0])
        ce = tf.reduce_sum(ce) / num_ex_f
        self.add_loss(ce)
        total_loss = self.get_loss()
        self.register_var('loss', total_loss)

        ans = tf.argmax(y_gt, 1)
        correct = tf.equal(ans, tf.argmax(y_out, 1))
        top5_acc = tf.reduce_sum(tf.to_float(
            tf.nn.in_top_k(y_out, ans, 5))) / num_ex_f
        self.register_var('top5_acc', top5_acc)
        acc = tf.reduce_sum(tf.to_float(correct)) / num_ex_f
        self.register_var('acc', acc)
        return total_loss

    def build_optim(self, loss):
        learn_rate = self.get_option('learn_rate')
        lr_decay = self.get_option('learn_rate_decay')
        steps_decay = self.get_option('steps_per_lr_decay')
        learn_rate = tf.train.exponential_decay(
            learn_rate, self.global_step, steps_decay, lr_decay,
            staircase=True)
        self.register_var('learn_rate', learn_rate)
        num_ex = tf.shape(self.get_var('x'))[0]
        optimizer_typ = self.get_option('optimizer')
        if optimizer_typ == 'momentum':
            mom = self.get_option('momentum')
            optimizer = tf.train.MomentumOptimizer(learn_rate, momentum=mom)
        elif optimizer_typ == 'adam':
            eps = 1e-7
            optimizer = tf.train.AdamOptimizer(learn_rate, epsilon=eps)
        else:
            raise Exception('Unknown optimizer type: {}'.format(optimizer_typ))
        train_step = optimizer.minimize(loss, global_step=self.global_step)
        return train_step

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        return results
    pass


tfplus.nn.model.register('resnet_imagenet_wrapper', ResNetImageNetModelWrapper)
