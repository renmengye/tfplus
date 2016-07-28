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
import time
import resnet_imagenet_example

tfplus.init('Eval a 50-layer ResNet on ImageNet')
UID_PREFIX = 'resnet_imgnet_ex'
DATASET = 'imagenet'
MODEL_NAME = 'resnet_imagenet_example'
NUM_CLS = 1000

# Main options
tfplus.cmd_args.add('id', 'str', None)
tfplus.cmd_args.add('gpu', 'int', -1)
tfplus.cmd_args.add('results', 'str', '../results')
tfplus.cmd_args.add('logs', 'str', '../logs')
tfplus.cmd_args.add('localhost', 'str', 'http://localhost')
tfplus.cmd_args.add('restore_model', 'str', None)
tfplus.cmd_args.add('restore_logs', 'str', None)
tfplus.cmd_args.add('prefetch', 'bool', False)

opt = tfplus.cmd_args.make()

# Initialize logging/saving folder.
if opt['id'] is None:
    uid = tfplus.nn.model.gen_id(UID_PREFIX)
else:
    uid = opt['id']
logs_folder = os.path.join(opt['logs'], uid)
log = tfplus.utils.logger.get(os.path.join(logs_folder, 'raw'))
tfplus.utils.LogManager(logs_folder).register('raw', 'plain', 'Raw Logs')
results_folder = os.path.join(opt['results'], uid)

# Initialize data.
data = {}
for split in ['valid']:
    data[split] = tfplus.data.create_from_main(
        DATASET, split=split, mode=split)


def get_iter(split, batch_size=128, cycle=True, max_queue_size=20,
             num_threads=20):
    batch_iter = BatchIterator(
        num=data[split].get_size(), progress_bar=False, shuffle=True,
        batch_size=batch_size, cycle=cycle,
        get_fn=data[split].get_batch_idx)
    if opt['prefetch']:
        return ConcurrentBatchIterator(
            batch_iter, max_queue_size=max_queue_size,
            num_threads=num_threads)
    else:
        return batch_iter

# Keep running eval.
while True:
    # New graph every time.
    with tf.Graph().as_default():
        with tf.Session() as sess:
            tf.set_random_seed(1234)

            # Initialize model.
            model = (tfplus.nn.model.create_from_main(MODEL_NAME)
                     .set_gpu(opt['gpu'])
                     .set_folder(results_folder)
                     .restore_options_from(opt['restore_model'])
                     .build_loss_eval()
                     )
            if opt['restore_model'] is None:
                raise Exception('Need to specify restore path.')
            MAX_RETRY = 20
            retry = 0
            restore_success = False
            while not restore_success and retry < MAX_RETRY:
                try:
                    model.restore_weights_from(sess, opt['restore_model'])
                    restore_success = True
                except Exception as e:
                    retry += 1
                    log.error(e)
                    log.info('Retry loading after 30 seconds')
                    time.sleep(30)

            if not restore_success:
                log.fatal('Restore failure')
            else:
                log.info('Restore success')

            # Initialize experiment.
            (
                tfplus.experiment.create_from_main('train')
                .set_session(sess)
                .set_model(model)
                .set_logs_folder(os.path.join(opt['logs'], uid))
                .set_localhost(opt['localhost'])
                .restore_logs(opt['restore_logs'])
                .add_csv_output('Loss', ['valid'])
                .add_csv_output('Top 1 Accuracy', ['valid'])
                .add_csv_output('Top 5 Accuracy', ['valid'])
                .add_runner(  # Full epoch evaluation on validation set.
                    tfplus.runner.create_from_main('average')
                    .set_name('valid')
                    .set_outputs(['loss', 'acc', 'top5_acc'])
                    .add_csv_listener('Loss', 'loss', 'valid')
                    .add_cmd_listener('Loss', 'loss')
                    .add_csv_listener('Top 1 Accuracy', 'acc', 'valid')
                    .add_cmd_listener('Top 1 Accuracy', 'acc')
                    .add_csv_listener('Top 5 Accuracy', 'top5_acc', 'valid')
                    .add_cmd_listener('Top 5 Accuracy', 'top5_acc')
                    .set_iter(get_iter('valid', batch_size=opt['batch_size'],
                                       cycle=False))
                    .set_phase_train(False)
                    .set_num_batch(500000)   # Just something more than needed.
                    .set_interval(1))
            ).run()
    time.sleep(1800)  # Sleep 30 minutes
