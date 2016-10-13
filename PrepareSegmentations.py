# -*- coding: utf-8 -*-

caffe_root   = '../../lib/crfasrnn/caffe/'
data_root    = '../../data/'
results_root = '../../results/'

framedir = data_root + 'OlympicSports/clips/'
patchdir = results_root + 'OlympicSports/patches/'
bbdir = data_root + 'OlympicSports/bboxes/'

image_typ = 'png'

THRES = 150

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
from scipy.misc import imsave, imresize, imread
import scipy.io
import matplotlib.pyplot as plt


def prepare_segmentation(subpath, bbpos, cliquenum):
    frame = imread(framedir + subpath)
    segmentation = np.zeros((frame.shape[0], frame.shape[1]))

    #Read patch from crfasrnn
    patchseg = imread(patchdir + subpath)
    #Threshold with global threshold
    patchseg = patchseg > THRES
    #Fill segmentation with number of current clique
    patchseg = cliquenum * patchseg

    try:
        segmentation[bbpos[2]:bbpos[4], bbpos[1]:bbpos[3]] = patchseg[:,:]
    except Exception:
        warnings.warn('patch not same size as in bb file', RuntimeWarning)
        continue

    #Fill not-segment with ignore value
    segmentation[segmentation == ] = -1

    print bbpos
    print '----------------'


def prepare_segmentations(cliquefile, bbfiles):
    ## LOAD CLIQUES FILE
    if not os.path.isfile(cliquefile):
        print "Cliquefile is not good, -.-"
        return

    cliquemat = scipy.io.loadmat(cliquefile)
    cliquemat = cliquemat['class_images']
    #Count of cliques:
    N = cliquemat.shape[1]

    ##LOAD BB FILES
    col_dtypes  = np.dtype([('frame', 'uint16'), ('top', 'uint16'), ('left', 'uint16'), ('width', 'uint16'), ('height', 'uint16')])
    bbfilelist = [os.path.basename(os.path.normpath(x)) for x in glob.glob(bbdir + bbfiles + '*bb')]
    bbmats = dict()
    patchlists = dict()
    for bbfile in bbfilelist:
        #Filename without extension
        bbbase = bbfile[:-3]
        bbmats[bbbase] = np.loadtxt(bbdir + bbfiles + bbfile, dtype = col_dtypes)
        patchlists[bbbase] = [os.path.basename(os.path.normpath(x)) for x in glob.glob(patchdir + bbfiles + bbbase + '/*png')]


    for n in range(0,N):
        cliquepatchpaths = cliquemat[0,n]
        for patchpath in cliquepatchpaths:
            patchpath = patchpath.split('/crops/',1)[1]
            video = patchpath.split('/')[1]
            print video
            idx = patchlists[video].index(patchpath[-10:])
            bbpos = bbmats[video][idx]
            prepare_segmentation(patchpath, bbpos, n)


def main(argv):
    cliquefile = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/data/cliques/cliques_leveldb/class_images_long_jump.mat'
    bbfiles = 'long_jump/'

    prepare_segmentations(cliquefile, bbfiles)


if __name__ == "__main__":
    main(sys.argv[1:])