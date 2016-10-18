# -*- coding: utf-8 -*-


##CONFIGURATION
data_root   = '../../data/OlympicSports/'
results_root   = '../../results/OlympicSports/'
image_root  = 'clips/'
bb_root     = 'bboxes/'
patch_root  = 'patches/'
patched_image_root = 'applied_patches/'

image_typ = 'png'
orig_typ  = 'jpg'
##

import sys, getopt
import warnings
import os
import numpy as np
import glob
from scipy.misc import imsave, imread
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def applypatches(inputdir, bbfile):
    #Read BoundingBox File
    col_dtypes  = np.dtype([('frame', 'uint16'), ('top', 'uint16'), ('left', 'uint16'), ('width', 'uint16'), ('height', 'uint16')])
    bbmat       = np.loadtxt(data_root + bb_root + bbfile, dtype = col_dtypes)

    #Get file list
    filelist = [os.path.basename(os.path.normpath(x)) for x in glob.glob(results_root + patch_root + inputdir + '*' + image_typ)]
    done_list = [os.path.basename(os.path.normpath(x)) for x in glob.glob(results_root + patched_image_root + inputdir + '*png')]
    filelist  = list(set(filelist) - set(done_list))
    filelist = np.sort(filelist)

    if not os.path.exists(results_root + patched_image_root + inputdir):
        os.makedirs(results_root + patched_image_root + inputdir)
    else:
        warnings.warn('The directory for the patched images already exists!', RuntimeWarning)

    #Iterate over BoundingBoxes
    for imf in filelist:
        #Read image
        #imf = filelist[bb[0]-1]
        im      = imread(data_root + image_root + inputdir + imf[:-len(image_typ)] + orig_typ)
        patch   = imread(results_root + patch_root + inputdir + imf)
        overlay = np.zeros((im.shape[0], im.shape[1]))

        #Get bounding box
        idx = filter(lambda x: x.isdigit(), imf)
        idx = int(idx)
        try:
            bb = bbmat[idx]
        except Exception:
            warnings.warn('More patches than bboxes', RuntimeWarning)
            continue

        #overlay[bb[2]:bb[4], bb[1]:bb[3], :] = 255-patch[:,:,np.newaxis]
        try:
            overlay[bb[2]:bb[4], bb[1]:bb[3]] = patch[:,:]
        except Exception:
            warnings.warn('patch not same size as in bb file', RuntimeWarning)
            continue

        #Save patch
        #imsave(results_root + patched_image_root + inputdir + imf[:-len(image_typ)] + 'png', overlay)
        fig = plt.figure(frameon=False)
        plt.imshow(im, interpolation='none')
        plt.imshow(overlay, cmap='jet', alpha=0.7, interpolation='none')
        fig.savefig(results_root + patched_image_root + inputdir + imf[:-len(image_typ)] + 'png')
        plt.close(fig)
        print imf


def main(argv):
   inputdir = 'high_jump/bvV-s0nZjgI_05042_05264/'
   bbfile   = 'high_jump/bvV-s0nZjgI_05042_05264.bb'
   applypatches(inputdir,bbfile)


if __name__ == "__main__":
    main(sys.argv[1:])