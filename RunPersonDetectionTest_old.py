# -*- coding: utf-8 -*-

caffe_root   = '../../lib/crfasrnn/caffe/'
model_root   = '../../model/'
data_root    = '../../data/'
results_root = '../../results/'

MODEL_FILE = model_root + 'TVG_CRFRNN_new_deploy.prototxt'
PRETRAINED = model_root + 'TVG_CRFRNN_COCO_VOC.caffemodel'


person_palette = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                  255,255,255,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

maxsize = 200

import sys, getopt
sys.path.insert(0, caffe_root + 'python')

# To save images without having a DISPLAY
import matplotlib as mpl
mpl.use('Agg')

import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
import matplotlib.pyplot as plt
from scipy.misc import imread
import warnings

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

    if not os.path.exists(data_root + patch_root + inputdir):
        os.makedirs(data_root + patch_root + inputdir)
    else:
        warnings.warn('The patch directory already exists!', RuntimeWarning)

    for filename in os.listdir(data_root + inputdir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            input_image = 255 * caffe.io.load_image(data_root + inputdir + filename)

            width = input_image.shape[0]
            height = input_image.shape[1]
            maxDim = max(width,height)

            image = PILImage.fromarray(np.uint8(input_image))

            if width > maxsize or height > maxsize:
                refSize = maxsize, maxsize
                image.thumbnail(refSize, PILImage.ANTIALIAS)

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
            # Get predictions
            segmentation = net.predict([im])

            segmentation2 = segmentation[0:cur_h, 0:cur_w]
            output_im = PILImage.fromarray(segmentation2)
            output_im.putpalette(person_palette)

            imsave(results_root + outputdir + inputdir + filename, output_im)

        else:
            continue

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