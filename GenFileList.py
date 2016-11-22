
from glob import glob
from random import shuffle
import numpy as np

sport = 'long_jump'

im_head = '/export/home/mfrank/data/OlympicSports/clips/'
label_head = '/export/home/mfrank/results/OlympicSports/segmentations/'

test_file = '/export/home/mfrank/src/ifp/fcn/' + sport + '/test.txt'
train_file = '/export/home/mfrank/src/ifp/fcn/' + sport + '/train.txt'

images = [x[len(im_head):-3] for x in glob(im_head + sport + '/*/*jpg')]
labels = [x[len(label_head):-3] for x in glob(label_head + sport + '/*/*npy')]

files = list(set(labels) & set(images))
shuffle(files)

pivot = np.floor(len(files)*0.7).astype('uint16')
train = files[:pivot]
test  = files[pivot:]

ftrain = file(train_file, 'w')
ftrain.writelines( "%s\n" % item for item in train )
ftrain.close()

ftest = file(test_file, 'w')
ftest.writelines( "%s\n" % item for item in test )
ftest.close()
