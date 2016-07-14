"""
Train a 50-layer ResNet on ImageNet.
Usage: python res_net_example.py --help
"""

import cslab_environ

import numpy as np
import os
import tensorflow as tf
import tfplus
import tfplus.data.mnist
import tfplus.data.cifar10
import tfplus.data.imagenet

from resnet_imagenet_model import ResNetImageNetModel

tfplus.init('Train a 50-layer ResNet on ImageNet')
UID_PREFIX = 'resnet_imgnet_ex'
DATASET = 'imagenet'
MODEL_NAME = 'resnet_imagenet_example'
NUM_CLS = 1000

# Main options
tfplus.cmd_args.add('gpu', 'int', -1)
tfplus.cmd_args.add('results', 'str', '../results')
tfplus.cmd_args.add('logs', 'str', '../logs')
tfplus.cmd_args.add('localhost', 'str', 'http://localhost')
tfplus.cmd_args.add('restore_model', 'str', None)
tfplus.cmd_args.add('restore_logs', 'str', None)
tfplus.cmd_args.add('batch_size', 'int', 128)
tfplus.cmd_args.add('prefetch', 'bool', False)

# Model options
tfplus.cmd_args.add('inp_height', 'int', 256)
tfplus.cmd_args.add('inp_width', 'int', 256)
tfplus.cmd_args.add('inp_depth', 'int', 3)
tfplus.cmd_args.add('inp_shrink', 'int', 32)
tfplus.cmd_args.add('layers', 'list<int>', [3, 4, 6, 3])
tfplus.cmd_args.add('strides', 'list<int>', [1, 2, 2, 2])
tfplus.cmd_args.add('channels', 'list<int>', [256, 256, 512, 1024, 2048])
tfplus.cmd_args.add('shortcut', 'str', 'projection')
tfplus.cmd_args.add('learn_rate', 'float', 0.01)
tfplus.cmd_args.add('learn_rate_decay', 'float', 0.1)
tfplus.cmd_args.add('steps_per_lr_decay', 'int', 160000)
tfplus.cmd_args.add('momentum', 'float', 0.9)
tfplus.cmd_args.add('wd', 'float', 1e-4)


class ResNetImageNetModelWrapper(tfplus.nn.Model):

    def __init__(self, name='resnet_imagenet_example'):
        super(ResNetImageNetModelWrapper, self).__init__(name=name)
        self.register_option('inp_height')
        self.register_option('inp_width')
        self.register_option('inp_depth')
        self.register_option('inp_shrink')
        self.register_option('layers')
        self.register_option('strides')
        self.register_option('channels')
        self.register_option('bottleneck')
        self.register_option('shortcut')
        self.register_option('learn_rate')
        self.register_option('learn_rate_decay')
        self.register_option('steps_per_lr_decay')
        self.register_option('momentum')
        self.register_option('wd')
        pass

    def init_default_options(self):
        self.set_default_option('bottleneck', True)

    def build_input(self):
        inp_height = self.get_option('inp_height')
        inp_width = self.get_option('inp_width')
        inp_depth = self.get_option('inp_depth')
        x = self.add_input_var(
            'x', [None, inp_height, inp_width, inp_depth], 'float')
        x_id = tf.identity(x)
        self.register_var('x_id', x_id)
        y_gt = self.add_input_var('y_gt', [None, NUM_CLS], 'float')
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
        inp_shrink = self.get_option('inp_shrink')
        layers = self.get_option('layers')
        strides = self.get_option('strides')
        channels = self.get_option('channels')
        bottleneck = self.get_option('bottleneck')
        shortcut = self.get_option('shortcut')
        wd = self.get_option('wd')
        phase_train = self.get_input_var('phase_train')

        self.rnd_trans = tfplus.nn.ImageRandomTransform(
            padding=0,
            shrink=inp_shrink,  # 256 -> 224 (random crop)
            rnd_hflip=True,
            rnd_vflip=False,
            rnd_transpose=False,
            rnd_size=False)
        self.res_net = ResNetImageNetModel()
        self.res_net.set_all_options({
            'inp_height': inp_height - inp_shrink,
            'inp_width': inp_width - inp_shrink,
            'inp_depth': inp_depth,
            'layers': layers,
            'strides': strides,
            'channels': channels,
            'bottleneck': bottleneck,
            'shortcut': shortcut,
            'weight_decay': wd
        })
        pass

    def build(self, inp):
        self.lazy_init_var()
        x = inp['x']
        phase_train = inp['phase_train']
        # x = tf.Print(x, [tf.reduce_mean(x)])
        x = self.rnd_trans({'input': x, 'phase_train': phase_train})
        # x = tf.Print(x, [tf.reduce_mean(x)])
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
        # ce = tf.Print(ce, [tf.reduce_mean(ce), tf.reduce_max(y_out),
        #                    tf.reduce_min(y_out), 10.0])
        num_ex_f = tf.to_float(tf.shape(inp['x'])[0])
        ce = tf.reduce_sum(ce) / num_ex_f
        # ce = tf.Print(ce, [tf.reduce_mean(ce), 11.0])
        self.add_loss(ce)
        total_loss = self.get_loss()
        self.register_var('loss', total_loss)

        correct = tf.equal(tf.argmax(y_gt, 1), tf.argmax(y_out, 1))
        acc = tf.reduce_sum(tf.to_float(correct)) / num_ex_f
        self.register_var('acc', acc)
        return total_loss

    def build_optim(self, loss):
        learn_rate = self.get_option('learn_rate')
        lr_decay = self.get_option('learn_rate_decay')
        steps_decay = self.get_option('steps_per_lr_decay')
        mom = self.get_option('momentum')
        num_ex = tf.shape(self.get_var('x'))[0]
        learn_rate = tf.train.exponential_decay(
            learn_rate, self.global_step, steps_decay, lr_decay,
            staircase=True)
        self.register_var('learn_rate', learn_rate)
        # optimizer = tf.train.AdamOptimizer(learn_rate, epsilon=eps)
        optimizer = tf.train.MomentumOptimizer(learn_rate, momentum=mom)
        train_step = optimizer.minimize(loss, global_step=self.global_step)
        return train_step

    def get_save_var_dict(self):
        results = {}
        if self.has_var('step'):
            results['step'] = self.get_var('step')
        self.add_prefix_to(None, self.res_net.get_save_var_dict(), results)
        self.log.info('Save variable list:')
        [self.log.info((v[0], v[1].name)) for v in results.items()]
        return results

    def get_aux_var_dict(self):
        results = super(ConvNetExampleModel, self).get_aux_var_dict()
        self.log.info('Aux variable list:')
        [self.log.info(v) for v in results.items()]
        return results
    pass


tfplus.nn.model.register(MODEL_NAME, ResNetImageNetModelWrapper)


if __name__ == '__main__':
    opt = tfplus.cmd_args.make()

    # Initialize logging/saving folder.
    uid = tfplus.nn.model.gen_id(UID_PREFIX)
    logs_folder = os.path.join(opt['logs'], uid)
    log = tfplus.utils.logger.get(os.path.join(logs_folder, 'raw'))
    tfplus.utils.LogManager(logs_folder).register('raw', 'plain', 'Raw Logs')
    results_folder = os.path.join(opt['results'], uid)

    # Initialize session.
    sess = tf.Session()
    tf.set_random_seed(1234)

    # Initialize model.
    model = (tfplus.nn.model.create_from_main(MODEL_NAME)
             .set_gpu(opt['gpu'])
             .set_folder(results_folder)
             .restore_options_from(opt['restore_model'])
             .build_all()
             .init(sess)
             .restore_weights_aux_from(sess, opt['restore_model']))

    # Initialize data.
    def get_data(split, batch_size=128, cycle=True):
        dp = tfplus.data.create_from_main(DATASET, split=split).set_iter(
            batch_size=batch_size, cycle=cycle)
        if opt['prefetch']:
            return tfplus.data.ConcurrentDataProvider(dp, max_queue_size=30)
        else:
            return dp

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
     .add_csv_output('Learning Rate', ['train'])
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
        .set_data_provider(get_data('train', batch_size=opt['batch_size'],
                                    cycle=True))
        .set_phase_train(True)
        .set_num_batch(10)
        .set_interval(1))
     .add_runner(
        tfplus.runner.create_from_main('saver')
        .set_name('saver')
        .set_interval(1))    # Every 1000 steps
     .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('trainval')
        .set_outputs(['acc', 'learn_rate'])
        .add_csv_listener('Accuracy', 'acc', 'train')
        .add_cmd_listener('Accuracy', 'acc')
        .add_csv_listener('Learning Rate', 'learn_rate', 'train')
        .set_data_provider(get_data('train', batch_size=opt['batch_size'],
                                    cycle=True))
        .set_phase_train(False)
        .set_num_batch(10)
        .set_offset(100)
        .set_interval(20))     # Every 200 steps
     .add_runner(  # Full epoch evaluation on validation set.
        tfplus.runner.create_from_main('average')
        .set_name('valid')
        .set_outputs(['acc'])
        .add_csv_listener('Accuracy', 'acc', 'valid')
        .add_cmd_listener('Accuracy', 'acc')
        .set_data_provider(get_data('valid', batch_size=opt['batch_size'],
                                    cycle=True))
        .set_phase_train(False)
        .set_num_batch(50000 / opt['batch_size'])
        .set_offset(100)
        .set_interval(500))    # Every 5000 steps.
     .add_runner(
        tfplus.runner.create_from_main('basic')
        .set_name('plotter')
        .set_outputs(['x_trans'])
        .add_plot_listener('Input', {'x_trans': 'images'})
        .set_data_provider(get_data('valid', batch_size=10, cycle=True))
        .set_phase_train(False)
        .set_offset(100)       # Every 1000 steps.
        .set_interval(10))).run()
    pass
