import caffe
from caffe import layers as L, params as P
import numpy as np

import surgery

caffe.set_device(0)
caffe.set_mode_gpu()

n = caffe.NetSpec()

