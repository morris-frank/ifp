# -*- coding: utf-8 -*-


##CONFIGURATION
data_root   = '../../data/'
results_root   = '../../results/'
overlay_root = 'overlays/'

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

def applyoverlay(inputdir, overlaydir):
    filelist = [os.path.basename(os.path.normpath(x)) for x in glob.glob(overlaydir + '*png')]
    filelist = np.sort(filelist)

    if not os.path.exists(results_root + overlay_root + inputdir):
        os.makedirs(results_root + overlay_root + inputdir)
    else:
        warnings.warn('The directory for the patched images already exists!', RuntimeWarning)

    #Iterate over BoundingBoxes
    for imf in filelist:
        #Read image
        #imf = filelist[bb[0]-1]
        im      = imread(data_root + inputdir + imf[:-len(image_typ)] + orig_typ)
        overlay = imread(overlaydir + imf)

        #Save patch
        #imsave(results_root + overlay_root + inputdir + imf[:-len(image_typ)] + 'png', overlay)
        fig = plt.figure(frameon=False)
        plt.imshow(im, interpolation='none')
        plt.imshow(overlay, cmap='jet', alpha=0.7, interpolation='none')
        fig.savefig(results_root + overlay_root + inputdir + imf[:-len(image_typ)] + 'png')
        plt.close(fig)
        print imf


def main(argv):
   inputdir = 'test/'
   overlaydir   = '../../results/test/'
   applyoverlay(inputdir,overlaydir)


if __name__ == "__main__":
    main(sys.argv[1:])