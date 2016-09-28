"""
Train a 50-layer ResNet on ImageNet.
Usage: python res_net_example.py --help
"""

import cslab_environ

import numpy as np
import os
import tensorflow as tf
import tfplus
import tfplus.data.imagenet
from tfplus.utils import BatchIterator, ConcurrentBatchIterator
import resnet_imagenet_model_wrapper
import resnet_imagenet_model_multi_wrapper
from acc_listener import AccuracyListener

tfplus.init('Train a 50-layer ResNet on ImageNet')

# Main options
tfplus.cmd_args.add('id', 'str', None)
tfplus.cmd_args.add('gpu', 'int', -1)
tfplus.cmd_args.add('results', 'str', '../results')
tfplus.cmd_args.add('logs', 'str', '../logs')
tfplus.cmd_args.add('localhost', 'str', 'http://localhost')
tfplus.cmd_args.add('restore_model', 'str', None)
tfplus.cmd_args.add('restore_logs', 'str', None)
tfplus.cmd_args.add('batch_size', 'int', 128)
tfplus.cmd_args.add('num_replica', 'int', 1)
tfplus.cmd_args.add('num_worker', 'int', 10)
tfplus.cmd_args.add('prefetch', 'bool', False)
opt = tfplus.cmd_args.make()

# Initialize logging/saving folder.
if opt['id'] is None:
    uid = tfplus.nn.model.gen_id('resnet_imgnet_ex')
else:
    uid = opt['id']

logs_folder = os.path.join(opt['logs'], uid)
log = tfplus.utils.logger.get(os.path.join(logs_folder, 'raw'))
tfplus.utils.LogManager(logs_folder).register('raw', 'plain', 'Raw Logs')
results_folder = os.path.join(opt['results'], uid)

# Initialize session.
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                        log_device_placement=True))
tf.set_random_seed(1234)

# Initialize model.
if opt['num_replica'] > 1:
    model = tfplus.nn.model.create_from_main('resnet_imagenet_multi_wrapper',
                                             num_replica=opt['num_replica'])
else:
    model = tfplus.nn.model.create_from_main('resnet_imagenet_wrapper')
model = (
    model
    .set_gpu(opt['gpu'])
    .set_folder(results_folder)
    .restore_options_from(opt['restore_model'])
    .build_all()
)

if opt['restore_model'] is not None:
    model.restore_weights_aux_from(sess, opt['restore_model'])
else:
    model.init(sess)

# Initialize data.
data = {}
for split in ['train', 'valid']:
    data[split] = tfplus.data.create_from_main(
        'imagenet', split=split, mode=split)


def get_data(split, batch_size=opt['batch_size'], cycle=True, shuffle=True,
             max_queue_size=100,
             num_threads=opt['num_worker'], log_epoch=200,
             prefetch=opt['prefetch']):
    batch_iter = BatchIterator(
        num=data[split].get_size(), progress_bar=False, shuffle=shuffle,
        batch_size=batch_size, cycle=cycle,
        get_fn=data[split].get_batch_idx, log_epoch=log_epoch)
    if prefetch:
        batch_iter = ConcurrentBatchIterator(
            batch_iter, max_queue_size=max_queue_size,
            num_threads=num_threads, log_queue=log_epoch)
    return batch_iter

# Initialize experiment.
exp = (
    tfplus.experiment.create_from_main('train')

    .set_session(sess)
    .set_model(model)
    .set_logs_folder(os.path.join(opt['logs'], uid))
    .set_localhost(opt['localhost'])
    .restore_logs(opt['restore_logs'])

    .add_csv_output('Loss', ['train'])
    .add_csv_output('Top 1 Accuracy (Train)', ['top-1 acc'])
    .add_csv_output('Top 5 Accuracy (Train)', ['top-5 acc'])
    .add_csv_output('Top 1 Accuracy (Valid)', ['top-1 acc'])
    .add_csv_output('Top 5 Accuracy (Valid)', ['top-5 acc'])
    .add_csv_output('Step Time', ['train'])
    .add_csv_output('Learning Rate', ['train'])
    .add_plot_output('Input (Train)', 'thumbnail', max_num_col=5)
    .add_plot_output('Input (Valid)', 'thumbnail', max_num_col=5)

    .add_runner(
        tfplus.runner.create_from_main('basic')
        .add_output('orig_x')
        .set_name('plotter_train')
        .add_plot_listener('Input (Train)', {'orig_x': 'images'})
        .set_iter(get_data('train', batch_size=10, prefetch=False))
        .set_phase_train(True)
        .set_offset(0)       # Every 500 steps (10 min)
        .set_interval(50))

    .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('train')
        .add_output('loss')
        .add_output('train_step')
        .add_output('step_time')
        .add_csv_listener('Loss', 'loss', 'train')
        .add_csv_listener('Step Time', 'step_time', 'train')
        .add_cmd_listener('Step', 'step')
        .add_cmd_listener('Loss', 'loss')
        .add_cmd_listener('Step Time', 'step_time')
        .set_iter(get_data('train', num_threads=1, max_queue_size=10))
        .set_phase_train(True)
        .set_num_batch(10)
        .set_interval(1))

    .add_runner(
        tfplus.runner.create_from_main('saver')
        .set_name('saver')
        .set_interval(100))    # Every 1000 steps (20 min)

    .add_runner(
        tfplus.runner.create_from_main('average')
        .set_name('trainval')
        .add_output('acc')
        .add_output('top5_acc')
        .add_output('learn_rate')
        .add_csv_listener('Top 1 Accuracy (Train)', 'acc', 'top-1 acc')
        .add_cmd_listener('Top 1 Accuracy (Train)', 'acc')
        .add_csv_listener('Top 5 Accuracy (Train)', 'top5_acc', 'top-5 acc')
        .add_cmd_listener('Top 5 Accuracy (Train)', 'top5_acc')
        .add_csv_listener('Learning Rate', 'learn_rate', 'train')
        .set_iter(get_data('train', num_threads=1, max_queue_size=10))
        .set_phase_train(False)
        .set_num_batch(10)
        .set_offset(10)
        .set_interval(40))     # Every 400 steps (4 min)

)

exp.add_runner(  # Full epoch evaluation on validation set.
    tfplus.runner.create_from_main('eval')
    .set_name('valid')
    .set_outputs(['score_out'])
    .set_iter(get_data('valid', cycle=False, shuffle=False))
    .add_listener(AccuracyListener(
        top_k=1,
        filename=os.path.join(logs_folder, 'Top 1 Accuracy (Valid).csv'),
        label='top-1 acc'))
    .add_listener(AccuracyListener(
        top_k=5,
        filename=os.path.join(logs_folder, 'Top 5 Accuracy (Valid).csv'),
        label='top-5 acc'))
    .set_phase_train(False)
    .set_cycle(True)
    .set_offset(10)
    .set_interval(200)
)    # Every 2000 steps (100 min)
exp.run()
pass
