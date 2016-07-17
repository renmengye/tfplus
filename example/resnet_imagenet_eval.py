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
import time

import resnet_imagenet_example

tfplus.init('Eval a 50-layer ResNet on ImageNet')
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
tfplus.cmd_args.add('prefetch', 'bool', False)

if __name__ == '__main__':
    opt = tfplus.cmd_args.make()

    # Initialize logging/saving folder.
    uid = tfplus.nn.model.gen_id(UID_PREFIX)
    logs_folder = os.path.join(opt['logs'], uid)
    log = tfplus.utils.logger.get(os.path.join(logs_folder, 'raw'))
    tfplus.utils.LogManager(logs_folder).register('raw', 'plain', 'Raw Logs')
    results_folder = os.path.join(opt['results'], uid)

    # Initialize data.
    def get_data(split, batch_size=128, cycle=True, max_queue_size=10,
                 num_threads=10):
        dp = tfplus.data.create_from_main(
            DATASET, split=split, mode=split).set_iter(
            batch_size=batch_size, cycle=cycle)
        if opt['prefetch']:
            return tfplus.data.ConcurrentDataProvider(
                dp, max_queue_size=max_queue_size, num_threads=num_threads)
        else:
            return dp

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
                         .build_eval()
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
                (tfplus.experiment.create_from_main('train')
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
                    .set_data_provider(get_data('valid',
                                                batch_size=opt['batch_size'],
                                                cycle=False))
                    .set_phase_train(False)
                    .set_num_batch(500000)   # Just something more than needed.
                    .set_interval(1))
                 ).run()
        time.sleep(1800)  # Sleep 30 minutes
    pass
