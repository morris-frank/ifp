# -*- coding: utf-8 -*-


##CONFIGURATION
data_root    = '../../results/'
data_rel     = 'OlympicSports/patches/'
##

import sys, getopt

import warnings
from os.path import basename, normpath
import glob
import ApplyPatches


def main(argv):
    #sportslist = [basename(normpath(x)) for x in glob.glob(data_root + data_rel + '*')]
    sportslist = ['long_jump']
    for sport in sportslist:
        print "--------" + sport + "--------"
        vidlist = [basename(normpath(x)) for x in glob.glob(data_root + data_rel + sport + '/*')]
        for vidd in vidlist:
            print sport + ': ' + vidd
            ApplyPatches.applypatches(sport + '/' + vidd + '/', sport + '/' + vidd + '.bb')


if __name__ == "__main__":
    main(sys.argv[1:])