import cslab_environ

import numpy as np
import os
import cv2
import tensorflow as tf
import tfplus

from resnet_imagenet_model import ResNetImageNetModel
from resnet_imagenet_model_wrapper import ResNetImageNetModelWrapper


def load_image(path, size=224):
    img = cv2.imread(path).astype('float32') / 255
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = cv2.resize(crop_img, (size, size))
    return resized_img


def print_prob(prob):
    pred = np.argsort(prob)[::-1]
    top1 = tfplus.data.synset.get_label(pred[0])
    print 'Top 1:'
    print top1, prob[pred[0]]
    top5 = [tfplus.data.synset.get_label(pred[i]) for i in xrange(5)]
    print 'Top 5:'
    for ii, tt in enumerate(top5):
        print tt, prob[pred[ii]]
    return top1

if __name__ == '__main__':
    folder = '/u/mren/third_party/tensorflow-resnet'
    img = load_image(os.path.join(folder, 'data/cat.jpg'))
    batch = img.reshape((1, 224, 224, 3))
    log = tfplus.utils.logger.get()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device('/cpu:0'):
                with log.verbose_level(2):
                    resnet = ResNetImageNetModel().set_all_options({
                        'inp_depth': 3,
                        'layers': [3, 8, 36, 3],
                        'strides': [1, 2, 2, 2],
                        'channels': [64, 256, 512, 1024, 2048],
                        'bottleneck': True,
                        'shortcut': 'projection',
                        'compatible': True,
                        'weight_decay': 1e-4,
                        'subtract_mean': True,
                        'trainable': False
                    })
                    inp_var = resnet.build_input()
                    out_var = resnet.build(inp_var)
            resnet.restore_weights_from(
                sess, '/ais/gobi4/mren/data/imagenet/res152')
            y_out = sess.run(out_var, feed_dict={
                             inp_var['x']: batch,
                             inp_var['phase_train']: False})
            print_prob(y_out[0])

    with tf.Graph().as_default():
        with tf.Session() as sess:
            with tf.device('/cpu:0'):
                with log.verbose_level(2):
                    wrapper = ResNetImageNetModelWrapper().set_all_options({
                        'inp_depth': 3,
                        'layers': [3, 8, 36, 3],
                        'strides': [1, 2, 2, 2],
                        'channels': [64, 256, 512, 1024, 2048],
                        'bottleneck': True,
                        'shortcut': 'projection',
                        'compatible': True,
                        'wd': 1e-4,
                        'subtract_mean': True,
                        'trainable': False
                    }).build_loss_eval().init(sess)
            wrapper.res_net.restore_weights_from(
                sess, '/ais/gobi4/mren/data/imagenet/res152')
            # wrapper.restore_weights_from(
            #     sess, '/ais/gobi4/mren/data/imagenet/res152_wrapper')
            y_out = sess.run(wrapper.get_var('y_out'), feed_dict={
                             wrapper.get_var('x'): batch,
                             wrapper.get_var('phase_train'): False})
            print_prob(y_out[0])
            wrapper.set_folder('/ais/gobi4/mren/data/imagenet/res152_wrapper')
            wrapper.save(sess)
