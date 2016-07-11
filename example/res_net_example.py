"""
Train a simple ResNet on CIFAR10.
Usage: python res_net_example.py --help
"""

import cslab_environ

import numpy as np
import os
import tensorflow as tf
import tfplus
import tfplus.data.mnist
import tfplus.data.cifar10

tfplus.init('Train a simple ResNet on CIFAR10')

# Main options
tfplus.cmd_args.add('gpu', 'int', -1)
tfplus.cmd_args.add('results', 'str', '../results')
tfplus.cmd_args.add('logs', 'str', '../logs')
tfplus.cmd_args.add('localhost', 'str', 'http://localhost')
tfplus.cmd_args.add('restore_model', 'str', None)
tfplus.cmd_args.add('restore_logs', 'str', None)

# Model options
tfplus.cmd_args.add('inp_height', 'int', 32)
tfplus.cmd_args.add('inp_width', 'int', 32)
tfplus.cmd_args.add('inp_depth', 'int', 3)
tfplus.cmd_args.add('layers', 'list<int>', [9, 9, 9])
tfplus.cmd_args.add('strides', 'list<int>', [1, 2, 2])
tfplus.cmd_args.add('channels', 'list<int>', [16, 16, 32, 64])
tfplus.cmd_args.add('bottleneck', 'bool', False)


class ResNetExampleModel(tfplus.nn.Model):

    def __init__(self, name='res_net_ex'):
        super(ResNetExampleModel, self).__init__(name=name)
        self.register_option('inp_height')
        self.register_option('inp_width')
        self.register_option('inp_depth')
        self.register_option('layers')
        self.register_option('strides')
        self.register_option('channels')
        self.register_option('bottleneck')
        pass

    def init_default_options(self):
        self.set_default_option('learn_rate', 1e-3)
        self.set_default_option('adam_eps', 1e-7)
        self.set_default_option('wd', None)
        pass

    def build_input(self):
        self.init_default_options()
        inp_height = self.get_option('inp_height')
        inp_width = self.get_option('inp_width')
        inp_depth = self.get_option('inp_depth')
        x = self.add_input_var(
            'x', [None, inp_height, inp_width, inp_depth], 'float')
        x_id = tf.identity(x)
        self.register_var('x_id', x_id)
        y_gt = self.add_input_var('y_gt', [None, 10], 'float')
        phase_train = self.add_input_var('phase_train', None, 'bool')
        return {
            'x': x,
            'y_gt': y_gt,
            'phase_train': phase_train
        }

    def init_var(self):
        inp_height = self.get_option('inp_height')
        inp_width = self.get_option('inp_width')
        inp_depth = self.get_option('inp_depth')
        layers = self.get_option('layers')
        strides = self.get_option('strides')
        channels = self.get_option('channels')
        bottleneck = self.get_option('bottleneck')
        wd = self.get_option('wd')
        phase_train = self.get_input_var('phase_train')

        init_channel = channels[0]
        self.w1 = tf.Variable(tf.truncated_normal_initializer(
            stddev=0.01)([7, 7, inp_depth, init_channel]), name='w1')
        self.b1 = tf.Variable(tf.truncated_normal_initializer(
            stddev=0.01)([init_channel]), name='b1')
        self.res_net = tfplus.nn.ResNet(layers=layers,
                                        bottleneck=bottleneck,
                                        channels=channels,
                                        strides=strides)
        cnn_dim = channels[-1]
        mlp_dims = [cnn_dim] + [10]
        mlp_act = [tf.nn.softmax]
        self.mlp = tfplus.nn.MLP(mlp_dims, mlp_act)
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['x']
        phase_train = inp['phase_train']
        channels = self.get_option('channels')
        init_channel = channels[0]

        h1 = tfplus.nn.Conv2D(self.w1, stride=2)(x) + self.b1
        bn1 = tfplus.nn.BatchNorm(init_channel)
        h1 = bn1({'input': h1, 'phase_train': phase_train})
        hn = self.res_net({'input': h1, 'phase_train': phase_train})
        hn = tfplus.nn.AvgPool(4)(hn)
        cnn_dim = channels[-1]
        hn = tf.reshape(hn, [-1, cnn_dim])
        y_out = self.mlp({'input': hn, 'phase_train': phase_train})
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

        correct = tf.equal(tf.argmax(y_gt, 1), tf.argmax(y_out, 1))
        acc = tf.reduce_sum(tf.to_float(correct)) / num_ex_f
        self.register_var('acc', acc)
        return total_loss

    def build_optim(self, loss):
        eps = self.get_option('adam_eps')
        learn_rate = self.get_option('learn_rate')
        optimizer = tf.train.AdamOptimizer(learn_rate, epsilon=eps)
        train_step = optimizer.minimize(loss, global_step=self.global_step)
        return train_step

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        self.add_prefix_to(
            'res_net', self.res_net.get_save_var_dict(), results)
        self.add_prefix_to('mlp', self.mlp.get_save_var_dict(), results)
        self.log.info('Save variable list:')
        [self.log.info((v[0], v[1].name)) for v in results.items()]
        return results

    def get_aux_var_dict(self):
        results = super(ConvNetExampleModel, self).get_aux_var_dict()
        self.log.info('Aux variable list:')
        [self.log.info(v) for v in results.items()]
        return results
    pass


tfplus.nn.model.register('res_net_example', ResNetExampleModel)


if __name__ == '__main__':
    opt = tfplus.cmd_args.make()

    # Initialize logging/saving folder.
    uid = tfplus.nn.model.gen_id('res_net_ex')
    logs_folder = os.path.join(opt['logs'], uid)
    log = tfplus.utils.logger.get(os.path.join(logs_folder, 'raw'))
    tfplus.utils.LogManager(logs_folder).register('raw', 'plain', 'Raw Logs')
    results_folder = os.path.join(opt['results'], uid)

    # Initialize session.
    sess = tf.Session()

    # Initialize model.
    model = (tfplus.nn.model.create_from_main('res_net_example')
             .set_gpu(opt['gpu'])
             .set_folder(results_folder)
             .restore_options_from(opt['restore_model'])
             .build_all()
             .init(sess)
             .restore_weights_aux_from(sess, opt['restore_model']))

    # Initialize data.
    data = {}
    for split in ['train', 'valid']:
        data[split] = tfplus.data.create_from_main('cifar10', split=split)
        pass

    # Initialize experiment.
    (tfplus.experiment.create_from_main('train')
     .set_session(sess)
     .set_model(model)
     .set_logs_folder(os.path.join(opt['logs'], uid))
     .set_localhost(opt['localhost'])
     .restore_logs(opt['restore_logs'])
     .add_csv_output('Loss', ['train'])
     .add_csv_output('Accuracy', ['train', 'valid'])
     .add_csv_output('Step Time', ['train'])
     .add_plot_output('Input', 'thumbnail', max_num_col=5)
     .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('train')
        .set_outputs(['loss', 'train_step'])
        .add_csv_listener('Loss', 'loss', 'train')
        .add_csv_listener('Step Time', 'step_time', 'train')
        .add_cmd_listener('Step', 'step')
        .add_cmd_listener('Loss', 'loss')
        .add_cmd_listener('Step Time', 'step_time')
        .set_iter(data['train'].get_iter(batch_size=32, cycle=True))
        .set_phase_train(True)
        .set_num_batch(10)
        .set_interval(1))
     .add_runner(
        tfplus.runner.create_from_main('saver')
        .set_name('saver')
        .set_interval(10))
     .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('trainval')
        .set_outputs(['acc'])
        .add_csv_listener('Accuracy', 'acc', 'train')
        .add_cmd_listener('Accuracy', 'acc')
        .set_iter(data['train'].get_iter(batch_size=32, cycle=True))
        .set_phase_train(False)
        .set_num_batch(10)
        .set_interval(10))
     .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('valid')
        .set_outputs(['acc'])
        .add_csv_listener('Accuracy', 'acc', 'valid')
        .add_cmd_listener('Accuracy', 'acc')
        .set_iter(data['valid'].get_iter(batch_size=32, cycle=True))
        .set_phase_train(False)
        .set_num_batch(10)
        .set_interval(10))
     .add_runner(
        tfplus.runner.create_from_main('basic')
        .set_name('plotter')
        .set_outputs(['x_id'])
        .add_plot_listener('Input', {'x_id': 'images'})
        .set_iter(data['valid'].get_iter(batch_size=10, stagnant=True))
        .set_phase_train(False)
        .set_interval(10))).run()
    pass
