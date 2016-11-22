caffe_root = '../../lib/caffe/'
model_root = '../../model/'
data_root = '../../data/'
results_root = '../../results/'

import sys
import getopt
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

import score
import surgery
from scipy.misc import imsave

from os.path import basename

import site
site.addsitedir('./oylmpic_layer')

#import setproctitle

sport = 'long_jump'
gpu = 0

weights = model_root + 'fcn/' + sport + '/snap_iter_30000.train.caffemodel'
solverf = './fcn/' + sport + '/solver.prototxt'

caffe.set_device(gpu)
caffe.set_mode_gpu()
# caffe.set_mode_cpu()

solver = caffe.AdaGradSolver(solverf)
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'deconv' in k]
surgery.interp(solver.net, interp_layers)

for _ in range(25):
    solver.step(4000)
