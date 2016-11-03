import caffe
import numpy as np
from glob import glob
import random
from PIL import Image
from os.path import normpath, basename


class OlympicDataLayer(caffe.Layer):
    """

    self.label_dir = ""
    self.data_dir = ""
    """

    def setup(self, bottom, top):
        print 'Setting up the OlympicDataLayer...'

        self.top_names = ['data', 'label']

        # config
        params = eval(self.param_str)
        self.data_dir = params['data_dir']
        self.label_dir = params['label_dir']
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

        self.idx = 0
        # Get all segmentations available without the extension
        self.paths = [basename(normpath(x))[:-3]
                      for x in glob(self.label_dir + '*' + self.label_ext)]
        self.paths = np.sort(self.paths)

    def reshape(self, bottom, top):
        # load image + label image pair
        self.data = self.load_image(self.paths[self.idx])
        self.label = self.load_label(self.paths[self.idx])

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

    def load_image(self, name):
        """
        Load input image and preprocess for Caffe:
        - cast to float
        - switch channels RGB -> BGR
        - subtract mean
        - transpose to channel x height x width order
        """
        im = Image.open(self.data_dir + name + self.data_ext)
        in_ = np.array(im, dtype=np.float32)
        in_ = in_[:, :, ::-1]
        in_ -= self.mean
        in_ = in_.transpose((2, 0, 1))
        return in_

    def load_label(self, name):
        label = np.load(self.label_dir + name +
                        self.label_ext)  # .astype('uin16')
        return label
