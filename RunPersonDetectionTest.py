# -*- coding: utf-8 -*-

caffe_root   = '../../lib/crfasrnn/caffe/'
model_root   = '../../model/'
data_root    = '../../data/'
results_root = '../../results/'

MODEL_FILE = model_root + 'TVG_CRFRNN_new_deploy_unary.prototxt'
PRETRAINED = model_root + 'TVG_CRFRNN_COCO_VOC.caffemodel'

image_typ = 'png'

person_palette = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

maxsize = 200

import sys, getopt
sys.path.insert(0, caffe_root + 'python')

import os
import glob
import warnings

import caffe
import numpy as np

# To save images without having a DISPLAY
import matplotlib as mpl
mpl.use('Agg')
from PIL import Image as PILImage
from scipy.misc import imsave
import matplotlib.pyplot as plt

# Import helper functions
import assets

def run_persondetectiontest(inputdir, outputdir, gpudevice):
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
        caffe.set_mode_cpu()
        net = caffe.Segmenter(MODEL_FILE, PRETRAINED,False)

    net.blobs['data'].reshape(1, 3, maxsize, maxsize)
    net.reshape()

    print net.blobs['data'].data.shape

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1)) # height*width*channel -> channel*height*width
    mean_file = np.array([104,117,123])
    transformer.set_mean('data', mean_file) #### subtract mean ####
    transformer.set_raw_scale('data', 255) # pixel value range
    transformer.set_channel_swap('data', (2,1,0)) # RGB -> BGR

    if not os.path.exists(results_root + outputdir + inputdir):
        os.makedirs(results_root + outputdir + inputdir)
    else:
        warnings.warn('The result directory already exists!', RuntimeWarning)

    filelist = [os.path.basename(x) for x in glob.glob(data_root + inputdir + '*' + image_typ)]
    filelist = np.sort(filelist)

    for filename in filelist:
        print filename
        input_image = caffe.io.load_image(data_root + inputdir + filename)

        width = input_image.shape[0]
        height = input_image.shape[1]
        maxDim = max(width,height)

        im = PILImage.fromarray(np.uint8(input_image))
        im = np.array(im)

        # Pad as necessary
        cur_h, cur_w, cur_c = im.shape
        pad_h = maxsize - cur_h
        pad_w = maxsize - cur_w
        im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)

        #net.blobs['data'].data[...] = transformer.preprocess('data', im)
        # Get predictions
        segmentation = net.forward()
        #segmentation = net.predict([im])


        segmentation2 = segmentation['pred'][0, 15, 0:cur_h, 0:cur_w]
        #segmentation2 = segmentation[0, 15, 0:cur_h, 0:cur_w]

        print segmentation2.shape
        #output_im = PILImage.fromarray(segmentation2)
        #output_im.putpalette(person_palette)

        imsave(results_root + outputdir + inputdir + filename, segmentation2)


def main(argv):
    inputdir = 'OlympicSports/patches/high_jump/bvV-s0nZjgI_05042_05264/'
    outputdir = ''
    gpu_device = 0
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["idir=","odir="])
    except getopt.GetoptError:
        print 'crfasrnn_demo.py -i <inputdir> -o <outputdir> -gpu <gpu_device>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'crfasrnn_demo.py -i <inputdir> -o <outputdir> -gpu <gpu_device>'
            sys.exit()
        elif opt in ("-i", "--idir"):
            inputdir = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
        elif opt in ("-gpu", "--gpudevice"):
            gpu_device = arg
    print 'Input directory is "', inputdir
    print 'Output directory is "', outputdir
    print 'GPU_DEVICE is "', gpu_device
    run_persondetectiontest(inputdir,outputdir,gpu_device)


if __name__ == "__main__":
    main(sys.argv[1:])