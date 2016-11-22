# -*- coding: utf-8 -*-

data_root = '../../data/'
results_root = '../../results/'

segdir = results_root + 'OlympicSports/segmentations/'
downdir = results_root + 'OlympicSports/segmentations_fourth/'

FACTOR = 0.25

import sys
import os.path
import glob
import warnings

import numpy as np
from scipy.misc import imresize
from skimage.measure import block_reduce
from progress.bar import Bar

def maxpool_helper(box,axis=-1):
    flattened = box.reshape((box.shape[0], box.shape[1], -1)).astype('uint16') +  1
    maxidx = flattened.max() + 1
    return np.apply_along_axis(lambda x: np.bincount(x, minlength=maxidx).argmax()-1, axis=2, arr=flattened).astype('int16')


def downsample_segmentation(segmentation, blockwidth):
    return block_reduce(segmentation, block_size=(blockwidth, blockwidth), func=maxpool_helper, cval=-1)


def downsample_segmentations():
    for sportpath in glob.glob(segdir + '*'):
        print 'Processing videos for ' + sportpath[len(segdir):]
        for vidpath in glob.glob(sportpath + '/*'):
            segmentations =  glob.glob(vidpath + '/*npy')
            bar = Bar(vidpath[len(sportpath):], max=len(segmentations))
            for segpath in segmentations:
                segmentation = np.load(segpath)
                segmentation = downsample_segmentation(segmentation, int(1/FACTOR))
                if not os.path.exists(downdir + subpath[:-10]):
                    os.makedirs(downdir + subpath[:-10])
                np.save(downdir + subpath, segmentation)
                bar.next()
            print


def main(argv):
    downsample_segmentations()


if __name__ == "__main__":
    main(sys.argv[1:])
