"""
Train a simple ConvNet on MNIST.
Usage: python conv_net_example.py --help
"""

import numpy as np
import tensorflow as tf
import tfplus
import tfplus.data.mnist

tfplus.init('Train a simple ConvNet on MNIST')
tfplus.cmd_args.add('inp_height', 'int', 28)
tfplus.cmd_args.add('inp_width', 'int', 28)
tfplus.cmd_args.add('inp_depth', 'int', 1)
tfplus.cmd_args.add('cnn_filter_size', 'list<int>', [3, 3, 3, 3])
tfplus.cmd_args.add('cnn_depth', 'list<int>', [8, 16, 32, 64])
tfplus.cmd_args.add('cnn_pool', 'list<int>', [2, 2, 2, 2])


class ConvNetExampleModel(tfplus.nn.Model):

    def __init__(self):
        super(ConvNetExampleModel, self).__init__()
        self.register_option('inp_height')
        self.register_option('inp_width')
        self.register_option('inp_depth')
        self.register_option('cnn_filter_size')
        self.register_option('cnn_depth')
        self.register_option('cnn_pool')
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
        with tf.device(self.get_device_fn()):
            x = self.add_input_var(
                'x', [None, inp_height, inp_width, inp_depth], 'float')
            x_id = tf.identity(x)
            self.register_var('x_id', x_id)
            y_gt = self.add_input_var('y_gt', [None, 10], 'float')
            phase_train = self.add_input_var('phase_train', None, 'bool')
            pass
        return {
            'x': x,
            'y_gt': y_gt,
            'phase_train': phase_train
        }

    def init_var(self):
        inp_height = self.get_option('inp_height')
        inp_width = self.get_option('inp_width')
        inp_depth = self.get_option('inp_depth')
        cnn_filters = self.get_option('cnn_filter_size')
        cnn_channels = [inp_depth] + self.get_option('cnn_depth')
        cnn_pool = self.get_option('cnn_pool')
        wd = self.get_option('wd')
        phase_train = self.get_input_var('phase_train')
        cnn_act = [tf.nn.relu] * len(cnn_filters)
        cnn_use_bn = [True] * len(cnn_filters)
        cnn_subsample = np.array(cnn_pool).prod()
        cnn_dim = cnn_channels[-1]
        mlp_dim = [cnn_dim * 2 * 2, 100, 10]
        mlp_act = [tf.nn.relu, tf.nn.softmax]
        with tf.device(self.get_device_fn()):
            self.cnn = tfplus.nn.CNN(cnn_filters, cnn_channels, cnn_pool,
                                     cnn_act, cnn_use_bn, wd=wd)
            self.mlp = tfplus.nn.MLP(mlp_dim, mlp_act)
            global_step = tf.Variable(0.0)
            self.register_var('step', global_step)
            pass
        pass

    def build(self, inp):
        self.lazy_init_var()
        with tf.device(self.get_device_fn()):
            x = inp['x']
            phase_train = inp['phase_train']
            h_cnn = self.cnn({'input': x, 'phase_train': phase_train})
            self.register_var('h_cnn', h_cnn)
            cnn_dim = self.get_option('cnn_depth')[-1]
            h_cnn = tf.reshape(h_cnn, [-1, cnn_dim * 2 * 2])
            y_out = self.mlp({'input': h_cnn, 'phase_train': phase_train})
            pass
        return {
            'y_out': y_out
        }

    def build_loss_grad(self, inp, output):
        with tf.device(self.get_device_fn()):
            y_gt = inp['y_gt']
            y_out = output['y_out']
            ce = tfplus.nn.CE()({'y_gt': y_gt, 'y_out': y_out})
            num_ex_f = tf.to_float(tf.shape(inp['x'])[0])
            ce = tf.reduce_sum(ce) / num_ex_f
            self.add_loss(ce)
            learn_rate = self.get_option('learn_rate')
            total_loss = self.get_loss()
            self.register_var('loss', total_loss)
            eps = self.get_option('adam_eps')
            optimizer = tf.train.AdamOptimizer(learn_rate, epsilon=eps)
            global_step = self.get_var('step')
            train_step = optimizer.minimize(
                total_loss, global_step=global_step)
            self.register_var('train_step', train_step)
            correct = tf.equal(tf.argmax(y_gt, 1), tf.argmax(y_out, 1))
            acc = tf.reduce_sum(tf.to_float(correct)) / num_ex_f
            self.register_var('acc', acc)
            pass
        pass
    pass


tfplus.nn.model.register('conv_net_example', ConvNetExampleModel)


if __name__ == '__main__':
    # Initialize session.
    sess = tf.Session()

    # Initialize model.
    model = (tfplus.nn.model.create_from_main('conv_net_example')
             .set_name('conv_net_ex')).build_all().init(sess)

    # Initialize data.
    data = {}
    for split in ['train', 'valid']:
        data[split] = tfplus.data.create_from_main('mnist', split=split)
        pass

    # Initialize experiment.
    (tfplus.experiment.create_from_main('train')
     .set_session(sess)
     .set_model(model)
     .add_csv_output('Loss', ['train'])
     .add_csv_output('Accuracy', ['train', 'valid'])
     .add_csv_output('Step Time', ['train'])
     .add_plot_output('Input', 'thumbnail', cmap='binary', max_num_col=5)
     .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('train')
        .set_outputs(['loss', 'train_step'])
        .add_csv_listener('Loss', 'loss', 'train')
        .add_csv_listener('Step Time', 'step_time', 'train')
        .add_cmd_listener('Step', 'step')
        .add_cmd_listener('Loss', 'loss')
        .add_cmd_listener('Step Time', 'step_time')
        .set_iter(data['train'].get_iter(batch_size=100, cycle=True))
        .set_phase_train(True)
        .set_num_batch(100)
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
        .set_iter(data['train'].get_iter(batch_size=100, cycle=True))
        .set_phase_train(False)
        .set_num_batch(10)
        .set_interval(10))
     .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('valid')
        .set_outputs(['acc'])
        .add_csv_listener('Accuracy', 'acc', 'valid')
        .add_cmd_listener('Accuracy', 'acc')
        .set_iter(data['valid'].get_iter(batch_size=100, cycle=True))
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
