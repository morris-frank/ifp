import caffe
import numpy as np
from glob import glob
import random
from PIL import Image
from os.path import normpath, basename
from scipy.misc import imresize
from DownsampleSegmentations import downsample_segmentation


class OlympicDataLayer(caffe.Layer):
    im_factor = 1.0
    label_factor = 0.25
    im_head = '/export/home/mfrank/data/OlympicSports/clips/'
    label_head = '/export/home/mfrank/results/OlympicSports/segmentations/'

    def setup(self, bottom, top):
        print 'Setting up the OlympicDataLayer...'

        self.top_names = ['data', 'label']

        # config
        params = eval(self.param_str)
        self.path_file = params['path_file']
        self.mean = np.array(params['mean'])
        self.random = params.get('randomize', False)
        self.seed = params.get('seed', None)
        self.data_ext = params.get('data_ext', 'jpg')
        self.label_ext = params.get('label_ext', 'npy')

        # two tops: data and label
        if len(top) != 2:
            raise Exception("Need to define two tops: data and label.")
        # data layers have no bottoms
        if len(bottom) != 0:
            raise Exception("Do not define a bottom.")

        self.paths = open(self.path_file, 'r').read().splitlines()
        self.idx = 0

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.paths[self.idx])
        self.label = self.load_label(self.paths[self.idx])

        if np.min([self.data.shape[1], self.data.shape[2]]) < 340:
            self.data = imresize(self.data, 2.0).transpose((2, 0, 1))
            self.label = self.label.repeat(2, axis=1).repeat(2, axis=2)

        # reshape tops to fit (leading 1 is for batch dimension)
        top[0].reshape(1, *self.data.shape)
        top[1].reshape(1, *self.label.shape)

    def forward(self, bottom, top):
        # assign output
        top[0].data[...] = self.data
        top[1].data[...] = self.label

        # pick next input
        if self.random:
            self.idx = random.randint(0, len(self.paths) - 1)
        else:
            self.idx += 1
            if self.idx == len(self.paths):
                self.idx = 0

    def backward(self, top, propagate_down, bottom):
        pass

    def load_image(self, path):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.im_head + path + self.data_ext)
        if self.im_factor == 1:
            in_ = im
        else:
            in_ = imresize(im, self.im_factor)
        in_ = np.array(in_, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_label(self, path):
        label = np.load(self.label_head + path + self.label_ext)  # .astype('uin16')
        if self.label_factor != 1:
            label = downsample_segmentation(label, int(1/self.label_factor))
        label = label[np.newaxis, ...]
        return label
