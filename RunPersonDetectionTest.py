# -*- coding: utf-8 -*-

caffe_root   = '../../lib/crfasrnn/caffe/'
model_root   = '../../model/'
data_root    = '../../data/'
results_root = '../../results/'

MODEL_FILE = model_root + 'TVG_CRFRNN_new_deploy_unary.prototxt'
PRETRAINED = model_root + 'TVG_CRFRNN_COCO_VOC.caffemodel'

image_typ = 'png'

maxsize = 227

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
    image.thumbnail((maxsize,maxsize), PILImage.ANTIALIAS)
    image = np.array(image)

    mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

    # Rearrange channels to form BGR
    im = image[:,:,::-1]
    # Subtract mean
    im = im - reshaped_mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = maxsize - cur_h
    pad_w = maxsize - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
    im = im.transpose((2, 0, 1))
    return im, cur_h, cur_w


def run_persondetectiontest(inputdir, outputdir, gpudevice):

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
        net = caffe.Segmenter(MODEL_FILE, PRETRAINED,True)
    else:
        print 'We wanna use a GPU, is not given, ya know.'
        sys.exit(2)

    net.blobs['data'].reshape(1, 3, maxsize, maxsize)
    net.reshape()

    for filename in filelist:
        print filename
        input_image = 255 * caffe.io.load_image(data_root + inputdir + filename)
        width = input_image.shape[0]
        height = input_image.shape[1]
        im, cur_h, cur_w = preprocess_image(input_image)

        # Get predictions
        net.blobs['data'].data[...] = im
        segmentation = net.forward()
        output_im = segmentation[net.outputs[0]]
        output_im = segmentation['pred'][0, 15, 0:cur_h, 0:cur_w]

        maxDim = max(width,height)
        if maxDim > maxsize:
            output_im = imresize(output_im, (width, height))

        imsave(results_root + outputdir + inputdir + filename, output_im)


def main(argv):
    #inputdir = 'OlympicSports/patches/high_jump/bvV-s0nZjgI_05042_05264/'
    inputdir = 'test/'
    outputdir = ''
    gpu_device = 0

    print 'Input directory is "', inputdir
    print 'Output directory is "', outputdir
    print 'GPU_DEVICE is "', gpu_device
    run_persondetectiontest(inputdir,outputdir,gpu_device)


if __name__ == "__main__":
    main(sys.argv[1:])