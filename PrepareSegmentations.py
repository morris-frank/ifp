# -*- coding: utf-8 -*-

data_root = '../../data/'
results_root = '../../results/'

framedir = data_root + 'OlympicSports/clips/'
patchdir = results_root + 'OlympicSports/patches/'
bbdir = data_root + 'OlympicSports/bboxes/'
segdir = results_root + 'OlympicSports/segmentations/'

image_typ = 'png'

THRES = 150
IGNORE_VALUE = -1

import sys
import getopt
import os.path
import glob
import warnings

import caffe
import numpy as np

# To save images without having a DISPLAY
import matplotlib as mpl
mpl.use('Agg')
from PIL import Image as PILImage
from scipy.misc import imresize, imread
import scipy.io
import matplotlib.pyplot as plt


def prepare_segmentation(subpath, bbpos, cliquenum):
    frame = imread(framedir + subpath[:-3] + 'jpg')

    # Create segmentation image filled with ignore values
    segmentation = np.full(
        (frame.shape[0], frame.shape[1]), IGNORE_VALUE, dtype=np.int16)

    # Read patch from crfasrnn
    patchseg = imread(patchdir + subpath)
    # Threshold with global threshold
    patchseg = (patchseg > THRES).astype(np.int16, copy=False)
    # Fill segmentation with number of current clique
    patchseg[patchseg == True] = cliquenum

    try:
        segmentation[bbpos[2]:bbpos[4], bbpos[1]:bbpos[3]] = patchseg[:, :]
    except Exception:
        warnings.warn('patch not same size as in bb file', RuntimeWarning)

    if not os.path.exists(segdir + subpath[:-10]):
        os.makedirs(segdir + subpath[:-10])

    np.save(segdir + subpath[:-4], segmentation)


def prepare_segmentations(cliquefile, bbfiles):
    # LOAD CLIQUES FILE
    if not os.path.isfile(cliquefile):
        print "Cliquefile is not good, -.-"
        return

    cliquemat = scipy.io.loadmat(cliquefile)
    cliquemat = cliquemat['class_images']
    # Count of cliques:
    N = cliquemat.shape[1]

    # LOAD BB FILES
    col_dtypes = np.dtype([('frame', 'uint16'), ('top', 'uint16'),
                           ('left', 'uint16'), ('width', 'uint16'), ('height', 'uint16')])
    bbfilelist = [os.path.basename(os.path.normpath(x))
                  for x in glob.glob(bbdir + bbfiles + '*bb')]
    bbmats = dict()
    patchlists = dict()
    for bbfile in bbfilelist:
        # Filename without extension
        bbbase = bbfile[:-3]
        bbmats[bbbase] = np.loadtxt(bbdir + bbfiles + bbfile, dtype=col_dtypes)
        patchlists[bbbase] = [os.path.basename(os.path.normpath(
            x)) for x in glob.glob(patchdir + bbfiles + bbbase + '/*png')]

    for n in range(0, N):
        cliquepatchpaths = cliquemat[0, n]
        print n
        for patchpath in cliquepatchpaths:
            patchpath = patchpath.split('/crops/', 1)[1]
            video = patchpath.split('/')[1]
            idx = patchlists[video].index(patchpath[-10:])
            bbpos = bbmats[video][idx]
            prepare_segmentation(patchpath, bbpos, n)


def main(argv):
    cliquefile = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/data/cliques/cliques_leveldb/class_images_long_jump.mat'
    bbfiles = 'long_jump/'

    prepare_segmentations(cliquefile, bbfiles)


if __name__ == "__main__":
    main(sys.argv[1:])
