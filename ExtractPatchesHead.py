# -*- coding: utf-8 -*-


##CONFIGURATION
data_root   = '../../data/OlympicSports/'
image_root  = 'clips/'
bb_root     = 'bboxes/'
##

import sys, getopt

import warnings
from os.path import basename, normpath
import glob
import ExtractPatches


def main(argv):
    sportslist = [basename(normpath(x)) for x in glob.glob(data_root + bb_root + '*')]

    for sport in sportslist:
        print "--------" + sport + "--------"
        bblist = [basename(normpath(x)) for x in glob.glob(data_root + bb_root + sport + '/*')]
        for bbf in bblist:
            print sport + ': ' + bbf[:-3]
            ExtractPatches.extractpatches(sport + '/' + bbf[:-3] + '/', sport + '/' + bbf)


if __name__ == "__main__":
    main(sys.argv[1:])