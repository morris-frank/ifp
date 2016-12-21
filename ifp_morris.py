import sys
import getopt
import warnings
import os
import glob
import shelve
import random
import numpy as np
import scipy.io as sio
from scipy.misc import imsave, imread, imresize
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from progress.bar import Bar
import skimage.measure
from PIL import Image as PILImage


# CONFIGURATION
DATA_ROOT = '../../data/OlympicSports/'
RESULTS_ROOT = '../../results/OlympicSports/'

PROTO_HEAD = './fcn/'

CLIPS_ROOT = DATA_ROOT + 'clips/'
BB_ROOT =  DATA_ROOT + 'bboxes/'
PATCH_ROOT = DATA_ROOT + 'patches/'

CRF_PATCH_ROOT = RESULTS_ROOT + 'patches/'
CRF_PATCHED_IMAGE_ROOT = RESULTS_ROOT + 'applied_patches/'
SEGMENTATION_ROOT = RESULTS_ROOT + 'segmentations/'
SEGMENTATION_PATCHES_ROOT = RESULTS_ROOT + 'segmentation_patches/'
OVERLAY_ROOT = RESULTS_ROOT + 'overlays/'
FCN_OVERLAY_ROOT = RESULTS_ROOT + 'overlays_fcn/'
FCN_ROOT = RESULTS_ROOT + 'fcn/'
ACTDB = FCN_ROOT + 'activationsDB'

OUTPUT_DTYPE = 'png'
CLIP_DTYPE = 'jpg'
SEG_DTYPE = 'npy'

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


def extract_patches_from_segmentation(inputdir, bbfile):
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
        SEGMENTATION_ROOT + inputdir + '*' + SEG_DTYPE)]
    filelist = np.sort(filelist)

    if len(filelist) < len(bbmat):
        print len(filelist)
        print len(bbmat)
        raise RuntimeError(
            'The BoundingBox File has more rows than images exist!')

    if not os.path.exists(SEGMENTATION_PATCHES_ROOT + inputdir):
        os.makedirs(SEGMENTATION_PATCHES_ROOT + inputdir)
    else:
        warnings.warn('The patch directory already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    bar = Bar('extract_patches_from_segmentation:' + bbfile, max=len(bbmat))
    for bb in bbmat:
        # Read image
        imf = filelist[bb[0] - 1]
        im = np.load(SEGMENTATION_ROOT + inputdir + imf[:-len(CLIP_DTYPE)] + SEG_DTYPE).astype('int')

        # Save patch
        patch = im[bb[2]:bb[4], bb[1]:bb[3], :]
        np.save(SEGMENTATION_PATCHES_ROOT + inputdir +
               imf[:-len(CLIP_DTYPE)] + SEG_DTYPE, patch)
        bar.next()


def extract_patches_from_segmentation_for_sport(sports):
    """
        Runs the extract_patches_from_segmentation function for all videos found for the given
        sports (list!).
        Exp.:
        apply_patches_for_sport(['long_jump'])
    """
    for sport in sports:
        print "--------" + sport + "--------"
        vidlist = [os.path.basename(os.path.normpath(x)) for x in glob.glob(
        CLIPS_ROOT + sport + '/*')]
        for vidd in vidlist:
            print sport + ': ' + vidd
            if not os.path.exists(BB_ROOT + sport + '/' + vidd + '.bb'):
                print 'BB File doesn\' exist'
                continue
            extract_patches_from_segmentation(
            sport + '/' + vidd + '/', sport + '/' + vidd + '.bb')


def apply_patches_fcn(inputdir, bbfile):
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
        FCN_ROOT + inputdir + '*' + OUTPUT_DTYPE)]
    done_list = [os.path.basename(os.path.normpath(x)) for x in glob.glob(
        FCN_OVERLAY_ROOT + inputdir + '*' + OUTPUT_DTYPE)]
    filelist = list(set(filelist) - set(done_list))
    filelist = np.sort(filelist)

    if not os.path.exists(FCN_OVERLAY_ROOT + inputdir):
        os.makedirs(FCN_OVERLAY_ROOT + inputdir)
    else:
        warnings.warn(
            'The directory for the patched images already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    bar = Bar('apply_patches_fcn' + bbfile, max=len(filelist))
    db = shelve.open(ACTDB)
    for imf in filelist:
        im = imread(CLIPS_ROOT + inputdir +
                    imf[:-len(OUTPUT_DTYPE)] + CLIP_DTYPE)
        patch = imread(FCN_ROOT + inputdir + imf)
        overlay = np.zeros((im.shape[0], im.shape[1]))
        clique = db[inputdir + imf[:-3]]
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
            half_horiz = int(np.ceil(patch.shape[0] / 2))
            half_vert = int(np.ceil(patch.shape[1] / 2))
            pivot_horiz = int(np.floor((bb[4]+bb[2]) / 2))
            pivot_vert = int(np.floor((bb[3]+bb[1]) / 2))
            center_horiz = min(im.shape[0]-half_horiz, pivot_horiz)
            center_vert = min(im.shape[1]-half_vert, pivot_vert)
            try:
                overlay[center_horiz-half_horiz:center_horiz-half_horiz+patch.shape[0],
                        center_vert-half_vert:center_vert-half_vert+patch.shape[1]] = patch[:, :]
            except Exception:
                warnings.warn('still not fitting ... give up')
                continue

        apply_overlay(im, overlay, FCN_OVERLAY_ROOT + inputdir + imf, label=str(clique))
        bar.next()


def apply_patches_for_sport_fcn(sports):
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
            apply_patches_fcn(
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
        CRF_PATCH_ROOT + inputdir + '*' + OUTPUT_DTYPE)]
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
                    imf[:-len(OUTPUT_DTYPE)] + CLIP_DTYPE)
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
        plt.imshow(overlay, cmap='plasma', alpha=0.7, interpolation='nearest')
        plt.axis('off')
        fig.savefig(CRF_PATCHED_IMAGE_ROOT +
                    inputdir + imf[:-len(CLIP_DTYPE)] + OUTPUT_DTYPE,
                    bbox_inches='tight')
        plt.close(fig)
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
    # Read patch from crfasrnn
    patchseg = imread(CRF_PATCH_ROOT + subpath)
    # Threshold with global threshold
    patchseg = (patchseg > SEGMENTATION_THRESHOLD).astype(int, copy=False)
    # Fill segmentation with number of current clique
    patchseg[patchseg == True] = clique
    patchseg[patchseg == False] = IGNORE_VALUE


    if boundingbox:
        frame = imread(CLIPS_ROOT + subpath[:-3] + CLIP_DTYPE)

        # Create segmentation image filled with ignore values
        segmentation = np.full(
            (frame.shape[0], frame.shape[1]), IGNORE_VALUE, dtype=int)

        try:
            segmentation[boundingbox[2]:boundingbox[4],
            boundingbox[1]:boundingbox[3]] = patchseg[:, :]
        except Exception:
            warnings.warn('patch not same size as in bb file', RuntimeWarning)
    else:
        segmentation = patchseg


    if boundingbox:
        if not os.path.exists(SEGMENTATION_ROOT + subpath[:-10]):
            os.makedirs(SEGMENTATION_ROOT + subpath[:-10])
        np.save(SEGMENTATION_ROOT + subpath[:-4], segmentation)
    else:
        if not os.path.exists(SEGMENTATION_PATCHES_ROOT + subpath[:-10]):
            os.makedirs(SEGMENTATION_PATCHES_ROOT + subpath[:-10])
        np.save(SEGMENTATION_PATCHES_ROOT + subpath[:-4], segmentation)


def prepare_segmentations(cliquefile, sport, full=True):
    """
        Run prepare_segmentation for sport and its cliquefile.
        E.g:
        cliquefile = '/net/hciserver03/storage/mbautist/Desktop/workspace/cnn_similarities/data/cliques/cliques_leveldb/class_images_long_jump.mat'
        sport = 'long_jump'
    """
    if not os.path.isfile(cliquefile):
        print cliquefile
        print "Cliquefile is not good, -.-"
        return
    cliquemat = sio.loadmat(cliquefile)
    cliquemat = cliquemat['class_images']
    N = cliquemat.shape[1]

    if full:
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

    bar = Bar('prepare_segmentations for ' + sport, max=N)
    for n in range(0, N):
        cliquepatchpaths = cliquemat[0, n]
        for patchpath in cliquepatchpaths:
            patchpath = patchpath.split('/crops/', 1)[1]
            bbpos = False
            if full:
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


def pad_n_mv_into_im(image, overlay_, put_in_middle=False):
    """
        Pad the given overlay so it fits the size of the image
    """
    doverlay_x = int(image.shape[0]-overlay_.shape[0])
    doverlay_y = int(image.shape[1]-overlay_.shape[1])
    if doverlay_x == 0 and doverlay_y == 0:
        return overlay_
    doverlay_x2 = int(np.floor(doverlay_x/2))
    doverlay_y2 = int(np.floor(doverlay_y/2))
    if doverlay_x > 0:
        overlay_ = np.pad(overlay_, ((doverlay_x2, doverlay_x2), (0, 0)), mode='edge')
    elif doverlay_x < 0:
        if put_in_middle:
            overlay_ = overlay_[-doverlay_x2:doverlay_x2, :]
        else:
            overlay_ = overlay_[:doverlay_x, :]
    if doverlay_y > 0:
        overlay_ = np.pad(overlay_, ((0, 0), (doverlay_y2, doverlay_y2)), mode='edge')
    elif doverlay_y < 0:
        if put_in_middle:
            overlay_ = overlay_[:, -doverlay_y:doverlay_y]
        else:
            overlay_ = overlay_[:, :doverlay_y]
    return overlay_


def apply_overlay(image, overlay, path, label=''):
    """
        Overlay overlay onto image and add label as text
        and save to path (full path with extension!)
    """
    fig = plt.figure(frameon=False)
    plt.imshow(image, interpolation='none')
    plt.imshow(overlay, cmap='plasma', alpha=0.7, interpolation='none')
    if label != '':
        red_patch = mpatches.Patch(color='yellow', label=label)
        plt.legend(handles=[red_patch])
    fig.savefig(path)
    plt.close(fig)


def apply_overlaydir(inputdir, overlaydir):
    """
        Run apply_overlay for all images and overlays in the corresponding
        directories.
        E.g.:
        inputdir = 'test/'
        overlaydir = '../../results/test/'
        applyoverlaydir(inputdir, overlaydir)
    """
    filelist = [os.path.basename(os.path.normpath(x))
                for x in glob.glob(overlaydir + '*' + OUTPUT_DTYPE)]
    filelist = np.sort(filelist)
    if not os.path.exists(OVERLAY_ROOT + inputdir):
        os.makedirs(OVERLAY_ROOT + inputdir)
    else:
        warnings.warn(
            'The directory for the patched images already exists!', RuntimeWarning)

    # Iterate over BoundingBoxes
    bar = Bar(inputdir, max=len(filelist))
    for imf in filelist:
        im = imread(DATA_ROOT + inputdir + imf[:-len(OUTPUT_DTYPE)] + CLIP_DTYPE)
        overlay = imread(overlaydir + imf)
        path = OVERLAY_ROOT + inputdir + imf[:-len(CLIP_DTYPE)] + OUTPUT_DTYPE
        apply_overlay(im, overlay, path)
        bar.next()


def apply_overlayfcn(listfile, factor=1):
    """
        Apply all overlays for the FCN results and load clique numbers
        from activations database and add as labels, factor is how much smaller
        the fcn outputs are...
        E.g.
        listfile = 'train.txt'
        factor = 0.25
    """
    paths = open(listfile, 'r').read().splitlines()
    db = shelve.open(ACTDB)
    bar = Bar(list, max=len(paths))
    for path in paths:
        im = imread(CLIPS_ROOT + path + CLIP_DTYPE)
        idx = db[path]
        ov = imread(FCN_ROOT + path + OUTPUT_DTYPE)
        if factor != 1:
            ov = imresize(ov, float(factor))
        ov = pad_n_mv_into_im(im, ov, put_in_middle=True)
        resdir = FCN_ROOT + path[:-8]
        if not os.path.exists(resdir):
            os.makedirs(resdir)
        apply_overlay(im, ov, resdir + path[-8:] + OUTPUT_DTYPE, str(idx))
        bar.next()


def gen_test_train_files(sport, ratio=0.7):
    """
        Generate training and testing sets for a given sport
    """
    images = [x[len(CLIPS_ROOT):-len(CLIP_DTYPE)]
        for x in glob.glob(CLIPS_ROOT + sport + '/*/*' + CLIP_DTYPE)]
    labels = [x[len(SEGMENTATION_ROOT):-len(SEG_DTYPE)]
        for x in glob.glob(SEGMENTATION_ROOT + sport + '/*/*' + SEG_DTYPE)]

    files = list(set(labels) & set(images))
    random.shuffle(files)

    pivot = np.floor(len(files)*ratio).astype('uint16')
    train = files[:pivot]
    test  = files[pivot:]

    ftrain = file(PROTO_HEAD + sport + '/train.txt', 'w')
    ftrain.writelines( "%s\n" % item for item in train )
    ftrain.close()

    ftest = file(PROTO_HEAD + sport + '/test.txt', 'w')
    ftest.writelines( "%s\n" % item for item in test )
    ftest.close()


def gen_test_train_files_patches(sport, ratio=0.7):
    """
        Generate training and testing sets for a given sport
    """
    images = [x[len(PATCH_ROOT):-len(OUTPUT_DTYPE)]
        for x in glob.glob(PATCH_ROOT + sport + '/*/*' + OUTPUT_DTYPE)]
    labels = [x[len(SEGMENTATION_PATCHES_ROOT):-len(SEG_DTYPE)]
        for x in glob.glob(SEGMENTATION_PATCHES_ROOT + sport + '/*/*' + SEG_DTYPE)]

    files = list(set(labels) & set(images))
    random.shuffle(files)

    pivot = np.floor(len(files)*ratio).astype('uint16')
    train = files[:pivot]
    test  = files[pivot:]

    ftrain = file(PROTO_HEAD + sport + '/train.txt', 'w')
    ftrain.writelines( "%s\n" % item for item in train )
    ftrain.close()

    ftest = file(PROTO_HEAD + sport + '/test.txt', 'w')
    ftest.writelines( "%s\n" % item for item in test )
    ftest.close()


def loadim(path):
    """
        Loads an image and prepares it
    """
    im = PILImage.open(path, dtype=np.float32)
