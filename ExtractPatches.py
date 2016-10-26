# -*- coding: utf-8 -*-


# CONFIGURATION
data_root = '../../data/OlympicSports/'
image_root = 'clips/'
bb_root = 'bboxes/'
patch_root = 'patches/'

image_typ = 'jpg'
##

import sys
import getopt
import warnings
import os
import numpy as np
import glob
from scipy.misc import imsave, imread


def extractpatches(inputdir, bbfile):
    # Read BoundingBox File
    col_dtypes = np.dtype([('frame', 'uint16'), ('top', 'uint16'),
                           ('left', 'uint16'), ('width', 'uint16'), ('height', 'uint16')])
    bbmat = np.loadtxt(data_root + bb_root + bbfile, dtype=col_dtypes)

    # Get file list
    filelist = [os.path.basename(x) for x in glob.glob(
        data_root + image_root + inputdir + '*' + image_typ)]
    filelist = np.sort(filelist)

    if len(filelist) < len(bbmat):
        raise RuntimeError(
            'The BoundingBox File has more rows than images exist!')

    if not os.path.exists(data_root + patch_root + inputdir):
        os.makedirs(data_root + patch_root + inputdir)
    else:
        warnings.warn('The patch directory already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    for bb in bbmat:
        # Read image
        imf = filelist[bb[0] - 1]
        im = imread(data_root + image_root + inputdir + imf)

        # Save patch
        patch = im[bb[2]:bb[4], bb[1]:bb[3], :]
        imsave(data_root + patch_root + inputdir +
               imf[:-len(image_typ)] + 'png', patch)


def main(argv):
    inputdir = 'high_jump/bvV-s0nZjgI_05042_05264/'
    bbfile = 'high_jump/bvV-s0nZjgI_05042_05264.bb'

    try:
        opts, args = getopt.getopt(argv, "hi:b:", ["idir=", "bfile="])
    except getopt.GetoptError:
        print 'ExtractPatches.py -i <inputdir> -b <bbfile>'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'ExtractPatches.py -i <inputdir> -b <bbfile>'
            sys.exit()
        elif opt in ("-i", "--idir"):
            inputdir = arg
        elif opt in ("-b", "--bfile"):
            bbfile = arg
    print 'Input directory is "', inputdir
    print 'Output directory is "', outputdir

    extractpatches(inputdir, bbfile)


if __name__ == "__main__":
    main(sys.argv[1:])
