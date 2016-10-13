# -*- coding: utf-8 -*-
"""
This package contains code for the "CRF-RNN" semantic image segmentation method, published in the
ICCV 2015 paper Conditional Random Fields as Recurrent Neural Networks. Our software is built on
top of the Caffe deep learning library.

Contact:
Shuai Zheng (szheng@robots.ox.ac.uk), Sadeep Jayasumana (sadeep@robots.ox.ac.uk), Bernardino Romera-Paredes (bernard@robots.ox.ac.uk)

Supervisor:
Philip Torr (philip.torr@eng.ox.ac.uk)

For more information about CRF-RNN, please vist the project website http://crfasrnn.torr.vision.
"""

caffe_root = '../caffe/'
import sys, getopt
sys.path.insert(0, caffe_root + 'python')

import os
import cPickle
import logging
import numpy as np
import pandas as pd
from PIL import Image as PILImage
#import Image
import cStringIO as StringIO
import caffe
# To save images without having a DISPLAY
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imsave


def tic():
    #http://stackoverflow.com/questions/5849800/tic-toc-functions-analog-in-python
    #Homemade version of matlab tic and toc functions
    import time
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()

def toc():
    import time
    if 'startTime_for_tictoc' in globals():
        print "Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds."
    else:
        print "Toc: start time not set"


def getpallete(num_cls):
        # this function is to get the colormap for visualizing the segmentation mask
        n = num_cls
        pallete = [0]*(n*3)
        for j in xrange(0,n):
                lab = j
                pallete[j*3+0] = 0
                pallete[j*3+1] = 0
                pallete[j*3+2] = 0
                i = 0
                while (lab > 0):
                        pallete[j*3+0] |= (((lab >> 0) & 1) << (7-i))
                        pallete[j*3+1] |= (((lab >> 1) & 1) << (7-i))
                        pallete[j*3+2] |= (((lab >> 2) & 1) << (7-i))
                        i = i + 1
                        lab >>= 3
        return pallete



def run_crfasrnn(inputfile, outputfile, gpudevice):
    MODEL_FILE = '../../model/TVG_CRFRNN_new_deploy_unary.prototxt'
    PRETRAINED = '../../model/TVG_CRFRNN_COCO_VOC.caffemodel'
    IMAGE_FILE = inputfile

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
        tic()
        net = caffe.Segmenter(MODEL_FILE, PRETRAINED,True)
        toc()
    else:
        caffe.set_mode_cpu()
        tic()
        net = caffe.Segmenter(MODEL_FILE, PRETRAINED,False)
        toc()


    input_image = 255 * caffe.io.load_image(IMAGE_FILE)


    width = input_image.shape[0]
    height = input_image.shape[1]
    maxDim = max(width,height)

    image = PILImage.fromarray(np.uint8(input_image))
    image = np.array(image)

    pallete = getpallete(256)

    mean_vec = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    reshaped_mean_vec = mean_vec.reshape(1, 1, 3);

    # Rearrange channels to form BGR
    im = image[:,:,::-1]
    # Subtract mean
    im = im - reshaped_mean_vec

    # Pad as necessary
    cur_h, cur_w, cur_c = im.shape
    pad_h = 200 - cur_h
    pad_w = 200 - cur_w
    im = np.pad(im, pad_width=((0, pad_h), (0, pad_w), (0, 0)), mode = 'constant', constant_values = 0)
    # Get predictions
    segmentation = net.predict([im])
    segmentation2 = segmentation[0:cur_h, 0:cur_w]
    output_im = PILImage.fromarray(segmentation2)
    output_im.putpalette(pallete)

    imsave(outputfile, output_im)


def main(argv):
   inputfile = '../../data/test/I00189.png'
   outputfile = '../../results/test/I00189.png'
   gpu_device = 0
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
   except getopt.GetoptError:
      print 'crfasrnn_demo.py -i <inputfile> -o <outputfile> -gpu <gpu_device>'
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print 'crfasrnn_demo.py -i <inputfile> -o <outputfile> -gpu <gpu_device>'
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
      elif opt in ("-gpu", "--gpudevice"):
         gpu_device = arg
   print 'Input file is "', inputfile
   print 'Output file is "', outputfile
   print 'GPU_DEVICE is "', gpu_device
   run_crfasrnn(inputfile,outputfile,gpu_device)


if __name__ == "__main__":
    main(sys.argv[1:])
