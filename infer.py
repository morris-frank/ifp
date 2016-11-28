caffe_root = '../../lib/caffe/'
model_root = '../../model/'
data_root = '../../data/'
results_root = '../../results/'

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from PIL import Image
from scipy.misc import imsave, imread
import sys
import getopt
sys.path.insert(0, caffe_root + 'python')
import caffe
import os
import site
site.addsitedir('./oylmpic_layer')
from progress.bar import Bar
import shelve
from ApplyOverlay import applyoverlayfcn


def loadim(path):
    im = Image.open(path)
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:,:,::-1]
    in_ -= np.array((103.939, 116.779, 123.68))
    in_ = in_.transpose((2,0,1))
    return in_


def inferfile(net, path_file, im_head):
    print 'Reading ' + path_file
    paths = open(path_file, 'r').read().splitlines()
    if not os.path.exists(results_root + 'OlympicSports/fcn/'):
        os.makedirs(results_root + 'OlympicSports/fcn/')
    db = shelve.open(results_root + 'OlympicSports/fcn/activationsDB')

    bar = Bar(path_file, max=len(paths))
    for path in paths:
        in_ = loadim(im_head + path + 'jpg')
        net.blobs['data'].reshape(1, *in_.shape)
        net.blobs['data'].data[...] = in_
        net.forward()
        out = net.blobs['deconv2'].data[0]
        maxidx = np.argmax(np.sum(out, axis=(1,2)))
        db[path] = maxidx
        db.sync()
        maxim = out[maxidx, ...]
        resdir = results_root + 'OlympicSports/fcn/' + path[:-8]
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        imsave(resdir + path[-8:] + 'png', maxim)
        bar.next()
    db.close()


def main(argv):
    sport = 'long_jump'
    model = 'snap_iter_100000.caffemodel'
    #---
    weights = model_root + 'fcn/' + sport + '/' + model
    netf = './fcn/' + sport + '/deploy.prototxt'

    gpu = 1
    caffe.set_device(gpu)
    caffe.set_mode_gpu()

    net = caffe.Net(netf, weights, caffe.TEST)
    im_head = '/export/home/mfrank/data/OlympicSports/clips/'
    test_path_file = 'fcn/' + sport + '/test.txt'
    train_path_file = 'fcn/' + sport + '/train.txt'

    inferfile(net, test_path_file, im_head)
    applyoverlayfcn(test_path_file, factor=4)

    # inferfile(net, train_path_file, im_head)
    # applyoverlayfcn(train_path_file, factor=4)




if __name__ == "__main__":
    main(sys.argv[1:])
