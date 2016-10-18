# -*- coding: utf-8 -*-

caffe_root   = '../../lib/crfasrnn/caffe/'
model_root   = '../../model/'
data_root    = '../../data/'
results_root = '../../results/'

MODEL_FILE = './fcn/long_jump/deploy.prototxt'
PRETRAINED = model_root + 'snapshots/long_jump/snap_iter_30000.caffemodel'

image_typ = 'jpg'

sizes = [35, 65, 67, 97, 99, 129, 131, 161, 163, 193, 195, 225, 227, 257, 259, 289, 291, 321, 323, 353, 355, 385, 387, 417, 419, 449, 451, 481, 483, 513, 515, 545, 547, 577, 579, 609, 611, 641, 643, 673, 675, 705, 707, 737, 739, 769, 771, 801, 803, 833, 835, 865, 867, 897, 899, 929, 931, 961, 963, 993, 995, 1025, 1027, 1057, 1059, 1089, 1091, 1121, 1123, 1153, 1155, 1185, 1187, 1217, 1219, 1249, 1251, 1281, 1283, 1313, 1315, 1345, 1347, 1377, 1379, 1409, 1411, 1441, 1443, 1473, 1475, 1505, 1507, 1537, 1539, 1569, 1571, 1601, 1603, 1633, 1635, 1665, 1667, 1697, 1699, 1729, 1731, 1761, 1763, 1793, 1795, 1825, 1827, 1857, 1859, 1889, 1891, 1921, 1923, 1953, 1955, 1985, 1987, 2017]
sizes = [131, 163, 195, 227, 259, 291, 323, 355, 387, 419, 451, 483, 515, 547, 579, 611, 643, 675, 707, 739, 771, 803, 835, 867, 899, 931, 963, 995, 1027, 1059, 1091, 1123, 1155, 1187, 1219, 1251, 1283, 1315, 1347, 1379, 1411, 1443, 1475, 1507]
sizes = [227, 259, 291, 323, 355, 387, 419, 451, 483, 515, 547, 579, 611, 643, 675, 707, 739, 771, 803, 835, 867, 899, 931, 963, 995, 1027, 1059, 1091, 1123, 1155, 1187, 1219, 1251, 1283, 1315, 1347, 1379, 1411, 1443, 1475, 1507]

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

# Import helper functions
import assets

def preprocess_image(file):
    input_image = 255 * file

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

    # Rearrange channels to form BGR
    im = image[:,:,::-1]
    # Subtract mean
    im = im - reshaped_mean_vec

    # Make sure its uint8
    im = PILImage.fromarray(np.uint8(im))
    im = np.array(im)

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    maxdim = max([cur_h, cur_w])
    #Variant 2 -> no-square
    nextvalid_h = next(x for x in sizes if x > cur_h)
    nextvalid_w = next(x for x in sizes if x > cur_w)

    pad_h = nextvalid_h - cur_h
    pad_w = nextvalid_w - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
    im = im.transpose((2, 0, 1))
    return im, cur_h, cur_w


def run_fcntest(inputdir, outputdir, gpudevice):

    if not os.path.exists(results_root + outputdir + inputdir):
        os.makedirs(results_root + outputdir + inputdir)
    else:
        warnings.warn('The result directory already exists!', RuntimeWarning)

    filelist  = [os.path.basename(os.path.normpath(x)) for x in glob.glob(data_root + inputdir + '*' + image_typ)]
    done_list = [os.path.basename(os.path.normpath(x)) for x in glob.glob(results_root + inputdir + '*')]
    filelist  = list(set(filelist) - set(done_list))
    filelist  = np.sort(filelist)

    if len(filelist) == 0:
        print "filelist was empty"
        return

    if gpudevice >=0:
        #Do you have GPU device?
        has_gpu = 1
        #which gpu device is available?
        gpu_device=gpudevice#assume the first gpu device is available, e.g. Titan X
    else:
        has_gpu = 0

    if has_gpu==1:
        caffe.set_device(gpu_device)
        caffe.set_mode_gpu()
        net = caffe.Net(MODEL_FILE, PRETRAINED,caffe.TEST)
    else:
        print 'We wanna use a GPU, is not given, ya know.'
        sys.exit(2)

    for filename in filelist:
        print filename
        input_image = 255 * caffe.io.load_image(data_root + inputdir + filename)
        print input_image.shape
        width = input_image.shape[0]
        height = input_image.shape[1]
        im, cur_h, cur_w = preprocess_image(input_image)

        print im.shape

        net.blobs['data'].reshape(1, *im.shape)
        #net.reshape()

        # Get predictions
        net.blobs['data'].data[...] = im
        segmentation = net.forward()

        out = net.blobs['fc8-conv'].data[0].argmax(axis=0)
        print out.shape

        out = imresize(out, (im.shape[1], im.shape[2]))
        print out.shape
        imsave(results_root + outputdir + inputdir + filename[:-3] + 'png', out)


def main(argv):
    #inputdir = 'OlympicSports/patches/long_jump/bvV-s0nZjgI_05042_05264/'
    inputdir = 'test/'
    outputdir = ''
    gpu_device = 0

    run_fcntest(inputdir,outputdir,gpu_device)


if __name__ == "__main__":
    main(sys.argv[1:])