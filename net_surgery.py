
caffe_root = '../../lib/caffe/'
model_root = '../../model/'
data_root = '../../data/'
results_root = '../../results/'

import sys
import getopt
sys.path.insert(0, caffe_root + 'python')

import os.path
import glob
import warnings

import caffe
import numpy as np

# To save images without having a DISPLAY
import matplotlib as mpl
mpl.use('Agg')
from PIL import Image as PILImage
from scipy.misc import imsave, imresize
import matplotlib.pyplot as plt
import site
site.addsitedir('./oylmpic_layer')


def transplant(new_net, net, suffix=''):
    """
    Transfer weights by copying matching parameters, coercing parameters of
    incompatible shape, and dropping unmatched parameters.
    The coercion is useful to convert fully connected layers to their
    equivalent convolutional layers, since the weights are the same and only
    the shapes are different.  In particular, equivalent fully connected and
    convolution layers have shapes O x I and O x I x H x W respectively for O
    outputs channels, I input channels, H kernel height, and W kernel width.
    Both  `net` to `new_net` arguments must be instantiated `caffe.Net`s.
    """
    for p in net.params:
        p_new = p + suffix
        if p_new not in new_net.params:
            print 'dropping', p
            continue
        for i in range(len(net.params[p])):
            if i > (len(new_net.params[p_new]) - 1):
                print 'dropping', p, i
                break
            if net.params[p][i].data.shape != new_net.params[p_new][i].data.shape:
                print 'coercing', p, i, 'from', net.params[p][i].data.shape, 'to', new_net.params[p_new][i].data.shape
            else:
                print 'copying', p, ' -> ', p_new, i
            new_net.params[p_new][i].data.flat = net.params[p][i].data.flat
    return new_net


def pre_transplant(inp_proto, inp_model, fcn_proto, fcn_model):
    state = caffe.TRAIN
    net = caffe.Net(inp_proto, inp_model, state)

    new_net = caffe.Net(fcn_proto, inp_model, state)

    new_net = transplant(new_net, net, '-deconv')
    new_net.save(fcn_model)


def perform_surgery(inp_proto, inp_model, fcn_proto, fcn_model):
    state = caffe.TRAIN

    # Load the original network and extract the fully connected layers'
    # parameters.
    net = caffe.Net(inp_proto, inp_model, state)
    params = ['fc6', 'fc7_', 'fc8_output']

    #net.blobs['data'].reshape(1, 3, 67, 67)
    # net.reshape()

    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[
                      pr][1].data) for pr in params}
    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(fcn_proto, inp_model, state)
    params_full_conv = ['fc6-conv', 'fc7-conv', 'fc8-score']

    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[
                        pr][1].data) for pr in params_full_conv}
    for conv in params_full_conv:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

    for pr, pr_conv in zip(params, params_full_conv):
        print '{} = {}'.format(pr_conv, pr)
        conv_params[pr_conv][0].flat = fc_params[
            pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    print 'Finished unrolling.....'

    if not os.path.exists('/'.join(fcn_model.split('/')[:-1])):
        os.makedirs('/'.join(fcn_model.split('/')[:-1]))

    net_full_conv.save(fcn_model)


def main(argv):
    sport = 'high_jump'
    sport = 'long_jump'

    inp_proto = model_root + 'snapshots/' + sport + '/net_config/deploy.prototxt'
    fcn_proto = './fcn/' + sport + '/train.prototxt'

    inp_model = model_root + 'snapshots/' + sport + '/snap_iter_30000.caffemodel'
    fcn_model = model_root + 'fcn/' + sport + '/original.caffemodel'

    #pre_transplant(inp_proto, inp_model, fcn_proto, fcn_model)
    perform_surgery(inp_proto, inp_model, fcn_proto, fcn_model)


if __name__ == "__main__":
    main(sys.argv[1:])
