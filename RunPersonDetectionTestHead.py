# -*- coding: utf-8 -*-


# CONFIGURATION
data_root = '../../data/'
data_rel = 'OlympicSports/clips/'
##

import sys
import getopt

import warnings
from os.path import basename, normpath
import glob
import RunPersonDetectionTest


def main(argv):
    sportslist = [basename(normpath(x))
                  for x in glob.glob(data_root + data_rel + '*')]

    for sport in sportslist:
        print "--------" + sport + "--------"
        vidlist = [basename(normpath(x)) for x in glob.glob(
            data_root + data_rel + sport + '/*')]
        for vidd in vidlist:
            print sport + ': ' + vidd
            RunPersonDetectionTest.run_persondetectiontest(
                data_rel + sport + '/' + vidd + '/', 'wholeimages', 0)


if __name__ == "__main__":
    main(sys.argv[1:])
