import sys
sys.path.insert(0, '/pkgs/tensorflow-gpu-0.9.0')
from convert import print_prob, load_image, checkpoint_fn, meta_fn
import tensorflow as tf

NLAYERS = 152


sess = tf.Session()

new_saver = tf.train.import_meta_graph(meta_fn(NLAYERS))
new_saver.restore(sess, checkpoint_fn(NLAYERS))

graph = tf.get_default_graph()
# for op in graph.get_operations():
#     print op.name

print 'graph restored'


def load_image(fname):
    import cv2
    image = cv2.imread(fname).astype('float32') / 255
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0] - short_edge) / 2)
    xx = int((image.shape[1] - short_edge) / 2)
    image = image[yy:yy + short_edge, xx:xx + short_edge]
    image = cv2.resize(image, (224, 224))
    image = image[:, :, [2, 1, 0]]
    return image


def test_image():
    img = load_image("data/cat.jpg")
    batch = img.reshape((1, 224, 224, 3))
    feed_dict = {images: batch}
    prob = sess.run(prob_tensor, feed_dict=feed_dict)
    print_prob(prob[0])
    vl = ['scale5/block3/Relu:0']
    # vl = ['scale1/Relu:0', 'scale1/Conv2D:0', 'sub:0']
    for v in vl:
        vten = sess.run(graph.get_tensor_by_name(v), feed_dict=feed_dict)
        print v
        print vten
        print vten.shape
    # print sess.run(graph.get_tensor_by_name("scale1/Relu:0"),
    #                feed_dict=feed_dict)
    # print sess.run(graph.get_tensor_by_name("scale1/Conv2D:0"),
    #                feed_dict=feed_dict)
    # print sess.run(graph.get_tensor_by_name("sub:0"), feed_dict=feed_dict)
    pass


def get_layers(nlayer):
    if nlayer == 50:
        return [3, 4, 6, 3]
    elif nlayer == 101:
        return [3, 4, 23, 3]
    elif nlayer == 152:
        return [3, 8, 36, 3]


def build_convert_dict():
    """
    ----------------------------------------------------------------------------
    Look up table
    ----------------------------------------------------------------------------
    Tensorflow-ResNet           My code
    ----------------------------------------------------------------------------
    s1/weights                  conv1/w
    s1/gamma                    bn1/gamma
    s1/beta                     bn1/beta
    s1/moving_mean              bn1/ema_mean
    s1/moving_variance          bn1/ema_var
    ----------------------------------------------------------------------------
    s{n}/b1/shortcut/weights    stage_{n-2}/shortcut/w
    s{n}/b1/shortcut/beta       stage_{n-2}/shortcut/bn/beta
    s{n}/b1/shortcut/gamma      stage_{n-2}/shortcut/bn/gamma
    s{n}/b1/moving_mean         stage_{n-2}/shortcut/bn/ema_mean
    s{n}/b1/moving_variance     stage_{n-2}/shortcut/bn/ema_var
    ----------------------------------------------------------------------------
    s{n}/b{m}/{a,b,c}/weights   stage_{n-1}/layer_{m-1}/unit_{k}/w
    s{n}/b{m}/{a,b,c}/beta      stage_{n-1}/layer_{m-1}/unit_{k}/bn/beta
    s{n}/b{m}/{a,b,c}/gamma     stage_{n-1}/layer_{m-1}/unit_{k}/bn/gamma
    s{n}/b{m}/moving_mean       stage_{n-1}/layer_{m-1}/unit_{k}/bn/ema_mean
    s{n}/b{m}/moving_variance   stage_{n-1}/layer_{m-1}/unit_{k}/bn/ema_var
    ----------------------------------------------------------------------------
    fc/weights                  fc/w
    fc/biases                   fc/b
    ----------------------------------------------------------------------------
    """
    vd = {}
    vd['conv1/w'] = graph.get_tensor_by_name('scale1/weights:0')
    vd['bn1/gamma'] = graph.get_tensor_by_name('scale1/gamma:0')
    vd['bn1/beta'] = graph.get_tensor_by_name('scale1/beta:0')
    vd['bn1/ema_mean'] = graph.get_tensor_by_name('scale1/moving_mean:0')
    vd['bn1/ema_var'] = graph.get_tensor_by_name('scale1/moving_variance:0')
    layers_list = get_layers(NLAYERS)

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
                # print ss, ll, k
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


def save_another():
    vd = build_convert_dict()
    vl = []
    for k in sorted(vd.keys()):
        vl.append(vd[k])

    rr = sess.run(vl)

    for kk, k in enumerate(sorted(vd.keys())):
        print k, rr[kk].shape

    tf.train.Saver(vd).save(sess, 'res_{}.ckpt'.format(NLAYERS))


def main():
    test_image()
    pass

if __name__ == '__main__':
    main()
