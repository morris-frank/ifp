import sys
import getopt
import warnings
import os
import glob
import numpy as np
from scipy.misc import imsave, imread, imresize
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from progress.bar import Bar
import skimage.measure


# CONFIGURATION
DATA_ROOT = '../../data/OlympicSports/'
RESULTS_ROOT = '../../results/OlympicSports/'

CLIPS_ROOT = DATA_ROOT + 'clips/'
BB_ROOT =  DATA_ROOT + 'bboxes/'
PATCH_ROOT = DATA_ROOT + 'patches/'

CRF_PATCH_ROOT = RESULTS_ROOT + 'patches/'
CRF_PATCHED_IMAGE_ROOT = RESULTS_ROOT + 'applied_patches/'
SEGMENTATION_ROOT = RESULTS_ROOT + 'segmentations/'

OUTPUT_DTYPE = 'png'
CLIP_DTYPE = 'jpg'

IGNORE_VALUE = -1
SEGMENTATION_THRESHOLD = 150
##


def extract_patches(inputdir, bbfile):
    """
        Extracts all bounding boxes from the bbfile and extracts all the
        patches from the corresponding clip frames from the inputdir.
        E.g:
        inputdir = 'high_jump/bvV-s0nZjgI_05042_05264/'
        bbfile = 'high_jump/bvV-s0nZjgI_05042_05264.bb'
    """
    # Read BoundingBox File
    col_dtypes = np.dtype([('frame', 'uint16'), ('top', 'uint16'),
                           ('left', 'uint16'), ('width', 'uint16'), ('height', 'uint16')])
    bbmat = np.loadtxt(BB_ROOT + bbfile, dtype=col_dtypes)

    # Get file list
    filelist = [os.path.basename(x) for x in glob.glob(
        CLIPS_ROOT + inputdir + '*' + CLIP_DTYPE)]
    filelist = np.sort(filelist)

    if len(filelist) < len(bbmat):
        raise RuntimeError(
            'The BoundingBox File has more rows than images exist!')

    if not os.path.exists(PATCH_ROOT + inputdir):
        os.makedirs(PATCH_ROOT + inputdir)
    else:
        warnings.warn('The patch directory already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    bar = Bar('extract_patches:' + bbfile, max=len(bbmat))
    for bb in bbmat:
        # Read image
        imf = filelist[bb[0] - 1]
        im = imread(CLIPS_ROOT + inputdir + imf)

        # Save patch
        patch = im[bb[2]:bb[4], bb[1]:bb[3], :]
        imsave(PATCH_ROOT + inputdir +
               imf[:-len(CLIP_DTYPE)] + OUTPUT_DTYPE, patch)
        bar.next()


def extract_patches_for_sport(sports):
    """
        Runs the extract_patches function for all videos found for the given
        sports (list!).
        Exp.:
        apply_patches_for_sport([]'long_jump'])
    """
    for sport in sports:
        print "--------" + sport + "--------"
        vidlist = [os.path.basename(os.path.normpath(x)) for x in glob.glob(
        CLIPS_ROOT + sport + '/*')]
        for vidd in vidlist:
            print sport + ': ' + vidd
            extract_patches(
            sport + '/' + vidd + '/', sport + '/' + vidd + '.bb')


def apply_patches(inputdir, bbfile):
    """
        Loads all bounding boxes from file bbfile and
        applies all FCN patches already saved onto the frames found in inputdir.
        Exp.:
        inputdir = 'high_jump/bvV-s0nZjgI_05042_05264/'
        bbfile = 'high_jump/bvV-s0nZjgI_05042_05264.bb'
        apply_patches(inputdir, bbfile)
    """
    # Read BoundingBox File
    col_dtypes = np.dtype([('frame', 'uint16'), ('top', 'uint16'),
                           ('left', 'uint16'), ('width', 'uint16'), ('height', 'uint16')])
    bbmat = np.loadtxt(BB_ROOT + bbfile, dtype=col_dtypes)

    # Get file list
    filelist = [os.path.basename(os.path.normpath(x)) for x in glob.glob(
        CRF_PATCH_ROOT + inputdir + '*' + CLIP_DTYPE)]
    done_list = [os.path.basename(os.path.normpath(x)) for x in glob.glob(
        CRF_PATCHED_IMAGE_ROOT + inputdir + '*' + OUTPUT_DTYPE)]
    filelist = list(set(filelist) - set(done_list))
    filelist = np.sort(filelist)

    if not os.path.exists(CRF_PATCHED_IMAGE_ROOT + inputdir):
        os.makedirs(CRF_PATCHED_IMAGE_ROOT + inputdir)
    else:
        warnings.warn(
            'The directory for the patched images already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    bar = Bar('apply_patches' + bbfile, max=len(filelist))
    for imf in filelist:
        im = imread(CLIPS_ROOT + inputdir +
                    imf[:-len(CLIP_DTYPE)] + CLIP_DTYPE)
        patch = imread(CRF_PATCH_ROOT + inputdir + imf)
        overlay = np.zeros((im.shape[0], im.shape[1]))

        # Get bounding box
        idx = filter(lambda x: x.isdigit(), imf)
        idx = int(idx)
        try:
            bb = bbmat[idx]
        except Exception:
            warnings.warn('More patches than bboxes', RuntimeWarning)
            continue

        #overlay[bb[2]:bb[4], bb[1]:bb[3], :] = 255-patch[:,:,np.newaxis]
        try:
            overlay[bb[2]:bb[4], bb[1]:bb[3]] = patch[:, :]
        except Exception:
            warnings.warn('patch not same size as in bb file', RuntimeWarning)
            continue

        fig = plt.figure(frameon=False)
        plt.imshow(im, interpolation='none')
        plt.imshow(overlay, cmap='jet', alpha=0.7, interpolation='none')
        fig.savefig(CRF_PATCHED_IMAGE_ROOT +
                    inputdir + imf[:-len(CLIP_DTYPE)] + OUTPUT_DTYPE)
        plt.close(fig)
        print imf
        bar.next()


def apply_patches_for_sport(sports):
    """
        Runs the apply_patches function for all videos found for the given
        sports (list!).
        Exp.:
        apply_patches_for_sport([]'long_jump'])
    """
    for sport in sports:
        print "--------" + sport + "--------"
        vidlist = [os.path.basename(os.path.normpath(x)) for x in glob.glob(
        CRF_PATCH_ROOT + sport + '/*')]
        for vidd in vidlist:
            print sport + ': ' + vidd
            apply_patches(
            sport + '/' + vidd + '/', sport + '/' + vidd + '.bb')


def prepare_segmentation(subpath, boundingbox, clique):
    """
        Prepare the segmentation of a full frame from a CRF run patch and the
        corresponding bounding box.
        E.g:
        subpath = 'long_jump/sdfgosdfpksd/I0003.jpg'
        boundingbox = [x_0, x_1, y_0, y_1]
        clique = 244
    """
    frame = imread(CLIPS_ROOT + subpath[:-3] + CLIP_DTYPE)

    # Create segmentation image filled with ignore values
    segmentation = np.full(
        (frame.shape[0], frame.shape[1]), IGNORE_VALUE, dtype=int)

    # Read patch from crfasrnn
    patchseg = imread(CRF_PATCH_ROOT + subpath)
    # Threshold with global threshold
    patchseg = (patchseg > SEGMENTATION_THRESHOLD).astype(int, copy=False)
    # Fill segmentation with number of current clique
    patchseg[patchseg == True] = clique

    try:
        segmentation[boundingbox[2]:boundingbox[4],
                     boundingbox[1]:boundingbox[3]] = patchseg[:, :]
    except Exception:
        warnings.warn('patch not same size as in bb file', RuntimeWarning)

    if not os.path.exists(SEGMENTATION_ROOT + subpath[:-10]):
        os.makedirs(SEGMENTATION_ROOT + subpath[:-10])

    np.save(SEGMENTATION_ROOT + subpath[:-4], segmentation)


def prepare_segmentations(cliquefile, sport):
    """
        Run prepare_segmentation for sport and its cliquefile.
        E.g:
        cliquefile = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/data/cliques/cliques_leveldb/class_images_long_jump.mat'
        bbfiles = 'long_jump'
    """
    if not os.path.isfile(cliquefile):
        print cliquefile
        print "Cliquefile is not good, -.-"
        return
    cliquemat = scipy.io.loadmat(cliquefile)
    cliquemat = cliquemat['class_images']
    N = cliquemat.shape[1]

    col_dtypes = np.dtype([('frame', 'uint16'), ('top', 'uint16'),
                           ('left', 'uint16'), ('width', 'uint16'), ('height', 'uint16')])
    bbfilelist = [os.path.basename(os.path.normpath(x))
                  for x in glob.glob(BB_ROOT + sport + '/*bb')]
    bbmats = dict()
    patchlists = dict()
    for bbfile in bbfilelist:
        # Filename without extension
        bbbase = bbfile[:-3]
        bbmats[bbbase] = np.loadtxt(BB_ROOT + sport + '/' + bbfile, dtype=col_dtypes)
        patchlists[bbbase] = [os.path.basename(os.path.normpath(x))
            for x in glob.glob(CRF_PATCH_ROOT + sport + '/' + bbbase + '/*' + OUTPUT_DTYPE)]

    for n in range(0, N):
        cliquepatchpaths = cliquemat[0, n]
        bar = Bar(n, max=len(cliquepatchpaths))
        for patchpath in cliquepatchpaths:
            patchpath = patchpath.split('/crops/', 1)[1]
            video = patchpath.split('/')[1]
            idx = patchlists[video].index(patchpath[-10:])
            bbpos = bbmats[video][idx]
            prepare_segmentation(patchpath, bbpos, n)
            bar.next()


def maxpool_helper(box,axis=-1):
    """
        Helper function for downsample_segmentation
        no docs, sry
    """
    flattened = box.reshape((box.shape[0], box.shape[1], -1)).astype('uint16') +  1
    maxidx = flattened.max() + 1
    return np.apply_along_axis(lambda x: np.bincount(x, minlength=maxidx).argmax()-1, axis=2, arr=flattened).astype('int16')


def downsample_segmentation(segmentation, blockwidth):
    """
        Downsamples the image (see segmentation) with factor of blockwidth
    """
    return skimage.measure.block_reduce(segmentation, block_size=(blockwidth, blockwidth), func=maxpool_helper, cval=-1)


def downsample_segmentations(new_segmentation_root, factor):
    """
        Downsample all segmentations into the directory
        new_segmentation_root with factor factor
        E.g:
        new_segmentation_root = '~/results/OlympicSports/segmentations_fourth'
        factor = 0.2
    """
    for sport in glob.glob(SEGMENTATION_ROOT + '*'):
        print 'Processing videos for ' + sport[len(SEGMENTATION_ROOT):]
        for vid in glob.glob(sport + '/*'):
            segmentations =  glob.glob(vid + '/*npy')
            bar = Bar(vid[len(sport):], max=len(segmentations))
            for segpath in segmentations:
                segmentation = np.load(segpath)
                segmentation = downsample_segmentation(segmentation, int(1/factor))
                if not os.path.exists(new_segmentation_root + subpath[:-10]):
                    os.makedirs(new_segmentation_root + subpath[:-10])
                np.save(new_segmentation_root + subpath, segmentation)
                bar.next()
            print
