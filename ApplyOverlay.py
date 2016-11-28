# -*- coding: utf-8 -*-


# CONFIGURATION
data_root = '../../data/'
results_root = '../../results/'
overlay_root = 'overlays/'

fcn_root = results_root + 'OlympicSports/fcn/'
im_root = data_root + 'OlympicSports/clips/'
actdb = fcn_root + 'activationsDB'

image_typ = 'png'
orig_typ = 'jpg'

fcn_put_in_middle = False

image_factor = 0.25
##

import sys
import getopt
import warnings
import os
import numpy as np
import glob
from scipy.misc import imsave, imread, imresize
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from progress.bar import Bar
import shelve

# GLOBAL VARIABLES
font = ImageFont.load('./gohufont-14.pil')

def pad_n_mv_into_im(im,ov):
    dovx = int(im.shape[0]-ov.shape[0])
    dovy = int(im.shape[1]-ov.shape[1])
    if dovx == 0 and dovy == 0:
        return ov
    dovx2 = int(np.floor(dovx/2))
    dovy2 = int(np.floor(dovy/2))
    if dovx > 0:
        ov = np.pad(ov, ((dovx2, dovx2), (0,0)), mode='edge')
    elif dovx < 0:
        if fcn_put_in_middle:
            ov = ov[-dovx2:dovx2, :]
        else:
            ov = ov[:dovx, :]
    if dovy > 0:
        ov = np.pad(ov, ((0,0), (dovy2, dovy2)), mode='edge')
    elif dovy < 0:
        if fcn_put_in_middle:
            ov = ov[:, -dovy:dovy]
        else:
            ov = ov[:, :dovy]
    return ov

def applyoverlay(im, overlay, path):
    fig = plt.figure(frameon=False)
    plt.imshow(im, interpolation='none')
    plt.imshow(overlay, cmap='jet', alpha=0.7, interpolation='none')
    fig.savefig(path)
    plt.close(fig)


def applyoverlaydir(inputdir, overlaydir):
    filelist = [os.path.basename(os.path.normpath(x))
                for x in glob.glob(overlaydir + '*png')]
    filelist = np.sort(filelist)

    if not os.path.exists(results_root + overlay_root + inputdir):
        os.makedirs(results_root + overlay_root + inputdir)
    else:
        warnings.warn(
            'The directory for the patched images already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    bar = Bar(inputdir, max=len(filelist))
    for imf in filelist:
        im = imread(data_root + inputdir + imf[:-len(image_typ)] + orig_typ)
        overlay = imread(overlaydir + imf)
        applyoverlay(im, overlay, results_root + overlay_root + inputdir + imf[:-len(image_typ)] + 'png')
        bar.next()


def applyoverlayfcn(list, factor=1):
    paths = open(list, 'r').read().splitlines()
    db = shelve.open(actdb)
    bar = Bar(list, max=len(paths))
    for path in paths:
        im = imread(im_root + path + orig_typ)
        idx = db[path]
        im_ = Image.fromarray(im)
        draw = ImageDraw.Draw(im_)
        draw.text((10,10), str(idx), font=font)
        im = np.array(im_)
        ov = imread(fcn_root + path + image_typ)
        if factor != 1:
            ov = imresize(ov, float(factor))
        ov = pad_n_mv_into_im(im, ov)
        resdir = results_root + 'OlympicSports/fcn_overlays/' + path[:-8]
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        applyoverlay(im, ov, resdir + path[-8:] + 'png')
        bar.next()


def main(argv):
    inputdir = 'test/'
    overlaydir = '../../results/test/'
    applyoverlaydir(inputdir, overlaydir)


if __name__ == "__main__":
    main(sys.argv[1:])
