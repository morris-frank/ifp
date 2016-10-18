
caffe_root   = '../../lib/caffe/'
model_root   = '../../model/'
data_root    = '../../data/'
results_root = '../../results/'

import sys, getopt
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

def perform_surgery(inp_proto, inp_model, fcn_proto, fcn_model):
    # Load the original network and extract the fully connected layers' parameters.
    net = caffe.Net(inp_proto, inp_model, caffe.TRAIN)
    params = ['fc6', 'fc7_', 'fc8_output']

    #net.blobs['data'].reshape(1, 3, 67, 67)
    print '#2'
    #net.reshape()
    print '#1'

    # fc_params = {name: (weights, biases)}
    fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
    for fc in params:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

    # Load the fully convolutional network to transplant the parameters.
    net_full_conv = caffe.Net(fcn_proto, inp_model, caffe.TEST)
    params_full_conv = ['fc6-conv', 'fc7_-conv', 'fc8_output-conv']

    # conv_params = {name: (weights, biases)}
    conv_params = {pr: (net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data) for pr in params_full_conv}
    for conv in params_full_conv:
        print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)

    for pr, pr_conv in zip(params, params_full_conv):
        conv_params[pr_conv][0].flat = fc_params[pr][0].flat  # flat unrolls the arrays
        conv_params[pr_conv][1][...] = fc_params[pr][1]

    if not os.path.exists('/'.join(fcn_model.split('/')[:-1])):
        os.makedirs('/'.join(fcn_model.split('/')[:-1]))

    net_full_conv.save(fcn_model)


def main(argv):
    sport = 'high_jump'
    sport = 'long_jump'

    inp_proto = model_root + 'snapshots/' + sport + '/net_config/deploy.prototxt'
    fcn_proto = './fcn/' + sport + '/train_test.prototxt'

    inp_model = model_root + 'snapshots/' + sport + '/snap_iter_30000.caffemodel'
    fcn_model = model_root + 'fcn/' + sport + '/snap_iter_30000.train.caffemodel'

    perform_surgery(inp_proto, inp_model, fcn_proto, fcn_model)


if __name__ == "__main__":
    main(sys.argv[1:])