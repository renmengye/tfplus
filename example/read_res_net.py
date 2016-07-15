import cslab_environ

import numpy as np
import tensorflow as tf
import tfplus

from resnet_imagenet_model import ResNetImageNetModel

sess = tf.Session()
resnet = ResNetImageNetModel().set_all_options({
    'inp_depth': 3,
    'layers': [3, 8, 36, 3],
    'strides': [1, 2, 2, 2],
    'channels': [64, 256, 512, 1024, 2048],
    'bottleneck': True,
    'shortcut': 'projection',
    'compatible': True,
    'weight_decay': 1e-4,
    'subtract_mean': True
}).build_eval()

#.init(sess)

saver = tf.train.Saver(resnet.get_save_var_dict())
saver.restore(sess, '/u/mren/third_party/tensorflow-resnet/res_152.ckpt')

# vd = resnet.get_save_var_dict()
# vl = []
# for k in sorted(vd.keys()):
#     vl.append(vd[k])

# rr = sess.run(vl)
# for kk, k in enumerate(sorted(vd.keys())):
#     print k, rr[kk].shape

import skimage.io
import skimage.transform


def load_image(path, size=224):
    img = skimage.io.imread(path)
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy:yy + short_edge, xx:xx + short_edge]
    resized_img = skimage.transform.resize(crop_img, (size, size))
    resized_img = resized_img[:, :, [2, 1, 0]] * 255.
    return resized_img

# returns the top1 string


def print_prob(prob):
    # print prob
    pred = np.argsort(prob)[::-1]
    # Get top1 label
    top1 = tfplus.data.synset.get_label(pred[0])
    print "Top1: ", top1, prob[pred[0]]
    # Get top5 label
    top5 = [tfplus.data.synset.get_label(pred[i]) for i in xrange(5)]
    print "Top 5:"
    for ii, tt in enumerate(top5):
        print tt, prob[pred[ii]]
    return top1

img = load_image("/u/mren/third_party/tensorflow-resnet/data/cat.jpg")
batch = img.reshape((1, 224, 224, 3))
y_out = sess.run(resnet.get_var('_y_out'), feed_dict={
                 resnet.get_var('x'): batch,
                 resnet.get_var('phase_train'): False})

print_prob(y_out[0])
print sess.run(resnet.get_var('_relu0'), feed_dict={
    resnet.get_var('x'): batch,
    resnet.get_var('phase_train'): False})
# print sess.run(resnet.get_var('_conv0'), feed_dict={
#     resnet.get_var('x'): batch,
#     resnet.get_var('phase_train'): False})
# print sess.run(resnet.get_var('_minus0'), feed_dict={
#     resnet.get_var('x'): batch,
#     resnet.get_var('phase_train'): False})
