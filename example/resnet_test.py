"""
A suite of test to make sure the converted ResNet model is correct.
Annoying fact: GPU outputs pretty non-deterministic values everytime (due to
parallel implementation and floating point precision issue.)
"""

import sys
# sys.path.insert(0, '/pkgs/tensorflow-gpu-0.9.0')
sys.path.insert(0, '..')
import tensorflow as tf
import os
import numpy as np
import cv2
from synset import *
import tfplus


def checkpoint_fn(layers):
    return 'ResNet-L%d.ckpt' % layers


def meta_fn(layers):
    return 'ResNet-L%d.meta' % layers


def load_old_model(sess, nlayers, device='/cpu:0'):
    with tf.device(device):
        new_saver = tf.train.import_meta_graph(meta_fn(nlayers))
    new_saver.restore(sess, checkpoint_fn(nlayers))
    graph = tf.get_default_graph()
    prob_tensor = graph.get_tensor_by_name("prob:0")
    images = graph.get_tensor_by_name("images:0")
    return graph, images, prob_tensor


def load_new_model(sess, restore_path, nlayers, device='/cpu:0'):
    from resnet_imagenet_model import ResNetImageNetModel
    with tf.device(device):
        logger = tfplus.utils.logger.get()
        with logger.verbose_level(2):
            resnet = ResNetImageNetModel().set_all_options({
                'inp_depth': 3,
                'layers': get_layers(nlayers),
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
            out_var2 = resnet.build(inp_var)
    saver = tf.train.Saver(resnet.get_save_var_dict())
    saver.restore(sess, restore_path)
    return resnet, inp_var, out_var, out_var2


def load_wrapper_model(sess, restore_path, nlayers, device='/cpu:0'):
    from resnet_imagenet_model_wrapper import ResNetImageNetModelWrapper
    with tf.device(device):
        logger = tfplus.utils.logger.get()
        with logger.verbose_level(2):
            resnet = ResNetImageNetModelWrapper().set_all_options({
                'inp_depth': 3,
                'layers': get_layers(nlayers),
                'strides': [1, 2, 2, 2],
                'channels': [64, 256, 512, 1024, 2048],
                'bottleneck': True,
                'shortcut': 'projection',
                'compatible': True,
                'wd': 1e-4,
                'subtract_mean': True,
                'trainable': False
            })
            inp_var = resnet.build_input()
            out_var = resnet.build(inp_var)
    saver = tf.train.Saver(resnet.res_net.get_save_var_dict())
    saver.restore(sess, restore_path)
    return resnet.res_net, inp_var, out_var['y_out']


def get_layers(nlayer):
    if nlayer == 50:
        return [3, 4, 6, 3]
    elif nlayer == 101:
        return [3, 4, 23, 3]
    elif nlayer == 152:
        return [3, 8, 36, 3]


def build_convert_dict(graph, nlayers):
    """
    ---------------------------------------------------------------------------
    Look up table
    ---------------------------------------------------------------------------
    Tensorflow-ResNet           My code
    ---------------------------------------------------------------------------
    s1/weights                  conv1/w
    s1/gamma                    bn1/gamma
    s1/beta                     bn1/beta
    s1/moving_mean              bn1/ema_mean
    s1/moving_variance          bn1/ema_var
    ---------------------------------------------------------------------------
    s{n}/b1/shortcut/weights    stage_{n-2}/shortcut/w
    s{n}/b1/shortcut/beta       stage_{n-2}/shortcut/bn/beta
    s{n}/b1/shortcut/gamma      stage_{n-2}/shortcut/bn/gamma
    s{n}/b1/moving_mean         stage_{n-2}/shortcut/bn/ema_mean
    s{n}/b1/moving_variance     stage_{n-2}/shortcut/bn/ema_var
    ---------------------------------------------------------------------------
    s{n}/b{m}/{a,b,c}/weights   stage_{n-2}/layer_{m-1}/unit_{k}/w
    s{n}/b{m}/{a,b,c}/beta      stage_{n-2}/layer_{m-1}/unit_{k}/bn/beta
    s{n}/b{m}/{a,b,c}/gamma     stage_{n-2}/layer_{m-1}/unit_{k}/bn/gamma
    s{n}/b{m}/moving_mean       stage_{n-2}/layer_{m-1}/unit_{k}/bn/ema_mean
    s{n}/b{m}/moving_variance   stage_{n-2}/layer_{m-1}/unit_{k}/bn/ema_var
    ---------------------------------------------------------------------------
    fc/weights                  fc/w
    fc/biases                   fc/b
    ---------------------------------------------------------------------------
    """
    vd = {}
    vd['conv1/w'] = graph.get_tensor_by_name('scale1/weights:0')
    vd['bn1/gamma'] = graph.get_tensor_by_name('scale1/gamma:0')
    vd['bn1/beta'] = graph.get_tensor_by_name('scale1/beta:0')
    vd['bn1/ema_mean'] = graph.get_tensor_by_name('scale1/moving_mean:0')
    vd['bn1/ema_var'] = graph.get_tensor_by_name('scale1/moving_variance:0')
    layers_list = get_layers(nlayers)

    for ss in xrange(2, 6):
        vd['res_net/stage_{}/shortcut/w'.format(ss - 2)] = \
            graph.get_tensor_by_name(
            'scale{}/block1/shortcut/weights:0'.format(ss))
        vd['res_net/stage_{}/shortcut/bn/beta'.format(ss - 2)] = \
            graph.get_tensor_by_name(
            'scale{}/block1/shortcut/beta:0'.format(ss))
        vd['res_net/stage_{}/shortcut/bn/gamma'.format(ss - 2)] = \
            graph.get_tensor_by_name(
            'scale{}/block1/shortcut/gamma:0'.format(ss))
        vd['res_net/stage_{}/shortcut/bn/ema_mean'.format(ss - 2)] = \
            graph.get_tensor_by_name(
            'scale{}/block1/shortcut/moving_mean:0'.format(ss))
        vd['res_net/stage_{}/shortcut/bn/ema_var'.format(ss - 2)] = \
            graph.get_tensor_by_name(
            'scale{}/block1/shortcut/moving_variance:0'.format(ss))
        for ll in xrange(layers_list[ss - 2]):
            for kk, k in enumerate(['a', 'b', 'c']):
                vd['res_net/stage_{}/layer_{}/unit_{}/w'.format(
                    ss - 2, ll, kk)] = \
                    graph.get_tensor_by_name(
                    'scale{}/block{}/{}/weights:0'.format(ss, ll + 1, k))
                vd['res_net/stage_{}/layer_{}/unit_{}/bn/beta'.format(
                    ss - 2, ll, kk)] = \
                    graph.get_tensor_by_name(
                    'scale{}/block{}/{}/beta:0'.format(ss, ll + 1, k))
                vd['res_net/stage_{}/layer_{}/unit_{}/bn/gamma'.format(
                    ss - 2, ll, kk)] = \
                    graph.get_tensor_by_name(
                    'scale{}/block{}/{}/gamma:0'.format(ss, ll + 1, k))
                vd['res_net/stage_{}/layer_{}/unit_{}/bn/ema_mean'.format(
                    ss - 2, ll, kk)] = \
                    graph.get_tensor_by_name(
                    'scale{}/block{}/{}/moving_mean:0'.format(ss, ll + 1, k))
                vd['res_net/stage_{}/layer_{}/unit_{}/bn/ema_var'.format(
                    ss - 2, ll, kk)] = \
                    graph.get_tensor_by_name(
                    'scale{}/block{}/{}/moving_variance:0'.format(ss, ll + 1, k))
    vd['fc/w'] = graph.get_tensor_by_name('fc/weights:0')
    vd['fc/b'] = graph.get_tensor_by_name('fc/biases:0')
    return vd


def save_convert_dict(sess, fname, vd):
    vl = []
    for k in sorted(vd.keys()):
        vl.append(vd[k])
    rr = sess.run(vl)
    # for kk, k in enumerate(sorted(vd.keys())):
    #     print k, rr[kk].shape
    tf.train.Saver(vd).save(sess, fname)
    pass


def load_image(fname):
    image = cv2.imread(fname).astype('float32') / 255
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    image = image[yy:yy + short_edge, xx:xx + short_edge]
    image = cv2.resize(image, (224, 224))
    image = image[:, :, [2, 1, 0]]
    return image


def test_hidden_old(sess, inp_var, out_var, img):
    batch = img.reshape((1, 224, 224, 3))
    feed_dict = {inp_var: batch}
    out_val = sess.run(out_var, feed_dict=feed_dict)
    print out_val
    pass


def test_hidden_new(sess, inp_var, out_var, img):
    batch = img.reshape((1, 224, 224, 3))
    feed_dict = {inp_var['x']: batch, inp_var['phase_train']: False}
    out_val = sess.run(out_var, feed_dict=feed_dict)
    print out_val
    pass


def print_prob(prob):
    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    top1 = synset[pred[0]]
    # Get top5 label
    top5 = [synset[pred[i]] for i in range(5)]
    for ii, tt in enumerate(top5):
        print '{:d} {:45s} {:.8f}'.format(ii, tt, prob[pred[ii]])
    return top1


def get_cpu_list():
    return ['ResizeBilinear', 'ResizeBilinearGrad', 'Mod', 'CumMin',
            'CumMinGrad', 'Hungarian', 'Reverse', 'SparseToDense',
            'BatchMatMul', 'Gather', 'Print', 'InTopK', 'TopKV2', 'Print']


def get_device_fn(device):
    """Choose device for different ops."""
    OPS_ON_CPU = set(get_cpu_list())

    def _device_fn(op):
        # Startswith "Save" is a hack...
        if op.type in OPS_ON_CPU or op.name.startswith('save'):
            return '/cpu:0'
        else:
            # Other ops will be placed on GPU if available, otherwise CPU.
            return device
    return _device_fn


def main():
    NLAYERS = 152
    # SAVE_FOLDER = '/ais/gobi4/mren/data'
    SAVE_FOLDER = '/dev/shm/models/res152'
    WRITE_GRAPH = False
    GRAPH_DIR1 = '/u/mren/logs'
    GRAPH_DIR2 = '/u/mren/logs2'
    # DEVICE = '/gpu:0'
    DEVICE = '/cpu:0'
    image_file = 'cat.jpg'
    image_data = load_image(image_file).reshape([1, 224, 224, 3])

    weights_file = os.path.join(SAVE_FOLDER, 'resnet_imagenet.ckpt-0'.format(NLAYERS))

    # Load old model
    output_old = {}
    old_vars = set()
    with tf.Graph().as_default():
        with tf.Session() as sess:
            graph, inp_var, out_var = load_old_model(
                sess, NLAYERS,
                device=get_device_fn(DEVICE))

            # Write graph.
            if WRITE_GRAPH:
                summary_writer = tf.train.SummaryWriter(
                    GRAPH_DIR2, graph_def=sess.graph_def)
                summary_writer.close()

            for vv in tf.all_variables():
                old_vars.add(vv.name)

            feed_dict = {inp_var: image_data}

            output_old['w1'] = sess.run(
                graph.get_tensor_by_name('scale1/weights:0'))
            output_old['bn1_beta'] = sess.run(
                graph.get_tensor_by_name('scale1/beta:0'))
            output_old['bn1_gamma'] = sess.run(
                graph.get_tensor_by_name('scale1/gamma:0'))
            output_old['bn1_mean'] = sess.run(
                graph.get_tensor_by_name('scale1/moving_mean:0'))
            output_old['bn1_var'] = sess.run(
                graph.get_tensor_by_name('scale1/moving_variance:0'))

            # Test output.
            print '-----------------------------------------------------------'
            print 'Old model pass 1'
            print_prob(sess.run(out_var, feed_dict=feed_dict)[0])
            print '-----------------------------------------------------------'

            # Test specific activation.
            output_old['sub'] = sess.run(
                graph.get_tensor_by_name('sub:0'), feed_dict=feed_dict)
            output_old['conv1'] = sess.run(graph.get_tensor_by_name(
                'scale1/Conv2D:0'), feed_dict=feed_dict)
            output_old['bn1'] = sess.run(graph.get_tensor_by_name(
                'scale1/batchnorm/add_1:0'), feed_dict=feed_dict)
            output_old['act2'] = sess.run(graph.get_tensor_by_name(
                'scale2/block3/Relu:0'), feed_dict=feed_dict)
            # output_old['shortcut3'] = sess.run(graph.get_tensor_by_name(
            #     'scale3/block1/shortcut/batchnorm/add_1:0'), feed_dict=feed_dict)
            for ll in xrange(8):
                output_old['act3_{}'.format(ll)] = sess.run(graph.get_tensor_by_name(
                    'scale3/block{}/Relu:0'.format(ll + 1)), feed_dict=feed_dict)
            output_old['act4'] = sess.run(graph.get_tensor_by_name(
                'scale4/block36/Relu:0'), feed_dict=feed_dict)
            output_old['act5'] = sess.run(graph.get_tensor_by_name(
                'scale5/block3/Relu:0'), feed_dict=feed_dict)

            # Check that there is no remain.
            vd = build_convert_dict(graph, NLAYERS)
            for vv in vd.itervalues():
                if vv.name in old_vars:
                    old_vars.remove(vv.name)

            for vv in list(old_vars):
                print 'Remaining', vv

            # save_convert_dict(sess, weights_file, vd)

            print '-----------------------------------------------------------'
            print 'Old model pass 2'
            print_prob(sess.run(out_var, feed_dict=feed_dict)[0])
            print '-----------------------------------------------------------'

    # Load new model
    output_new = {}
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model, inp_var, out_var, out_var2 = load_new_model(
                sess, weights_file, NLAYERS,
                device=get_device_fn(DEVICE))

            # Write graph.
            if WRITE_GRAPH:
                summary_writer = tf.train.SummaryWriter(
                    GRAPH_DIR2, graph_def=sess.graph_def)
                summary_writer.close()

            # Input is BGR.
            feed_dict = {inp_var['x']: image_data[:, :, :, [2, 1, 0]],
                         inp_var['phase_train']: False}

            # Test output.
            print '-----------------------------------------------------------'
            print 'New model pass 1'
            print_prob(sess.run(out_var, feed_dict=feed_dict)[0])
            print '-----------------------------------------------------------'

            output_new['w1'] = sess.run(model.conv1.w)
            output_new['bn1_beta'] = sess.run(model.bn1.beta)
            output_new['bn1_gamma'] = sess.run(model.bn1.gamma)
            output_new['bn1_mean'] = sess.run(
                model.bn1.get_save_var_dict()['ema_mean'])
            output_new['bn1_var'] = sess.run(
                model.bn1.get_save_var_dict()['ema_var'])

            # Test specific activation.
            output_new['sub'] = sess.run(
                model.get_var('x_sub'), feed_dict=feed_dict)
            output_new['conv1'] = sess.run(
                model.get_var('h_conv1'), feed_dict=feed_dict)
            output_new['bn1'] = sess.run(
                model.get_var('h_bn1'), feed_dict=feed_dict)
            output_new['act2'] = sess.run(model.res_net.get_var(
                'stage_0/layer_2/relu'), feed_dict=feed_dict)
            # output_new['shortcut3'] = sess.run(
            #     model.res_net.get_var('stage_1/shortcut'), feed_dict=feed_dict)
            for ll in xrange(8):
                output_new['act3_{}'.format(ll)] = sess.run(
                    model.res_net.get_var('stage_1/layer_{}/relu'.format(ll)),
                    feed_dict=feed_dict)
            output_new['act4'] = sess.run(model.res_net.get_var(
                'stage_2/layer_35/relu'), feed_dict=feed_dict)
            output_new['act5'] = sess.run(model.res_net.get_var(
                'stage_3/layer_2/relu'), feed_dict=feed_dict)

            print '-----------------------------------------------------------'
            print 'New model pass 2'
            print_prob(sess.run(out_var2, feed_dict=feed_dict)[0])
            print '-----------------------------------------------------------'

    output_wrapper = {}
    with tf.Graph().as_default():
        with tf.Session() as sess:
            model, inp_var, out_var = load_wrapper_model(
                sess, weights_file, NLAYERS,
                device=get_device_fn(DEVICE))

            # Write graph.
            if WRITE_GRAPH:
                summary_writer = tf.train.SummaryWriter(
                    GRAPH_DIR2, graph_def=sess.graph_def)
                summary_writer.close()

            # Input is BGR.
            feed_dict = {inp_var['x']: image_data[:, :, :, [2, 1, 0]],
                         inp_var['phase_train']: False}

            # Test output.
            print '-----------------------------------------------------------'
            print 'Wrapper model pass 1'
            print_prob(sess.run(out_var, feed_dict=feed_dict)[0])
            print '-----------------------------------------------------------'

            output_wrapper['w1'] = sess.run(model.conv1.w)
            output_wrapper['bn1_beta'] = sess.run(model.bn1.beta)
            output_wrapper['bn1_gamma'] = sess.run(model.bn1.gamma)
            output_wrapper['bn1_mean'] = sess.run(
                model.bn1.get_save_var_dict()['ema_mean'])
            output_wrapper['bn1_var'] = sess.run(
                model.bn1.get_save_var_dict()['ema_var'])

            # Test specific activation.
            output_wrapper['sub'] = sess.run(
                model.get_var('x_sub'), feed_dict=feed_dict)
            output_wrapper['conv1'] = sess.run(
                model.get_var('h_conv1'), feed_dict=feed_dict)
            output_wrapper['bn1'] = sess.run(
                model.get_var('h_bn1'), feed_dict=feed_dict)
            output_wrapper['act2'] = sess.run(model.res_net.get_var(
                'stage_0/layer_2/relu'), feed_dict=feed_dict)
            # output_wrapper['shortcut3'] = sess.run(
            #     model.res_net.get_var('stage_1/shortcut'), feed_dict=feed_dict)
            for ll in xrange(8):
                output_wrapper['act3_{}'.format(ll)] = sess.run(
                    model.res_net.get_var('stage_1/layer_{}/relu'.format(ll)),
                    feed_dict=feed_dict)
            output_wrapper['act4'] = sess.run(model.res_net.get_var(
                'stage_2/layer_35/relu'), feed_dict=feed_dict)
            output_wrapper['act5'] = sess.run(model.res_net.get_var(
                'stage_3/layer_2/relu'), feed_dict=feed_dict)

            print '-----------------------------------------------------------'
            print 'Wrapper model pass 2'
            print_prob(sess.run(out_var, feed_dict=feed_dict)[0])
    print '-----------------------------------------------------------'

    # Check all intermediate values.
    print '-----------------------------------------------------------'
    print 'Summary'
    print '{:15s}\t{:10s}\t{:10s}'.format('variable', 'diff', 'rel diff')
    print '-----------------------------------------------------------'
    for kk in sorted(output_old.keys()):
        diff = np.abs(output_old[kk] - output_new[kk]).mean()
        denom = np.abs(output_old[kk]).mean()
        print '{:15s}\t{:.8f}\t{:.8f}'.format(kk, diff, diff / denom)
    print '-----------------------------------------------------------'

    print '-----------------------------------------------------------'
    print 'Summary 2'
    print '{:15s}\t{:10s}\t{:10s}'.format('variable', 'diff', 'rel diff')
    print '-----------------------------------------------------------'
    for kk in sorted(output_old.keys()):
        diff = np.abs(output_old[kk] - output_wrapper[kk]).mean()
        denom = np.abs(output_old[kk]).mean()
        print '{:15s}\t{:.8f}\t{:.8f}'.format(kk, diff, diff / denom)
    print '-----------------------------------------------------------'

if __name__ == '__main__':
    main()
