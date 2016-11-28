import sys
import getopt
caffe_root = '../../lib/caffe/'
sys.path.insert(0, caffe_root + 'python')
import numpy as np
import caffe
from caffe import layers as L, params as P
from caffe.coord_map import crop
import site
site.addsitedir('./oylmpic_layer')

def conv(name, bottom, nout, ks=3, stride=1, pad=0, group=1, bias_val=0, weight_std=0.01):
    conv = L.Convolution(bottom, name=name, kernel_size=ks, group=group, stride=stride,
        num_output=nout, pad=pad,
        weight_filler=dict(type='gaussian', std=weight_std),
        bias_filler=dict(type='constant', value=bias_val),
        param=[dict(name=name+'_w', lr_mult=0, decay_mult=1), dict(name=name+'_b', lr_mult=0, decay_mult=0)])
    return conv


def upsample(name, bottom, nout, factor, weight_name):
    ks=2*factor-factor%2
    pad=int(np.ceil((factor-1)/2.0))
    deconv = L.Deconvolution(bottom, name=name,
        convolution_param=dict(kernel_size=ks, group=nout, stride=factor,
            num_output=nout, pad=pad,
            weight_filler=dict(type='bilinear'),
            bias_term=False),
        param=[dict(name=weight_name, lr_mult=0, decay_mult=1)])
    return deconv


def deconv(name, bottom, nout, factor, weight_std, bias_val):
    ks=2*factor-factor%2
    pad=int(np.ceil((factor-1)/2.0))
    deconv = L.Deconvolution(bottom, name=name,
        convolution_param=dict(kernel_size=ks, stride=factor,
            num_output=nout, pad=pad, bias_term=False,),
        param=[dict(lr_mult=0, decay_mult=0)])
    return deconv


def BuildBaseNet(sport, depth, split):
    n = caffe.NetSpec()

    pydata_params = dict(split=split, mean=(103.939, 116.779, 123.68))
    if split == 'train':
        pydata_params['path_file'] = '/export/home/mfrank/src/ifp/fcn/'+sport+'/train.txt'
    elif split == 'test':
        pydata_params['path_file'] = '/export/home/mfrank/src/ifp/fcn/'+sport+'/test.txt'

    if split == 'train' or split == 'test':
        n.data, n.label = L.Python(module='olympic_data_layer', layer='OlympicDataLayer',
            ntop=2, param_str=str(pydata_params))
    elif split == 'deploy':
        n.data = L.Input(name='input', shape=[dict(dim=[1,3,500,500])])

    n.conv1 = conv('conv1', n.data, 96, 11, 4)
    n.relu1 = L.ReLU(n.conv1, name='relu1', in_place=True)
    n.norm1 = L.LRN(n.relu1, name='norm1', local_size=5, alpha=0.0001, beta=0.75)
    n.pool1 = L.Pooling(n.norm1, name='pool1', kernel_size=3, stride=2, pool=P.Pooling.MAX)

    n.conv2 = conv('conv2', n.pool1, 256, 5, 1, 2, 2, 0.1)
    n.relu2 = L.ReLU(n.conv2, name='relu2', in_place=True)
    n.norm2 = L.LRN(n.relu2, name='norm2', local_size=5, alpha=0.0001, beta=0.75)
    n.pool2 = L.Pooling(n.norm2, name='pool2', kernel_size=3, stride=2, pool=P.Pooling.MAX)

    n.conv3 = conv('conv3', n.pool2, 384, 3, 1, 1)
    n.relu3 = L.ReLU(n.conv3, name='relu3', in_place=True)

    n.conv4 = conv('conv4', n.relu3, 384, 3, 1, 1, 2, 0.1)
    n.relu4 = L.ReLU(n.conv4, name='relu4', in_place=True)

    n.conv5 = conv('conv5', n.relu4, 256, 3, 1, 1, 2, 0.1)
    n.relu5 = L.ReLU(n.conv5, name='relu5', in_place=True)
    n.pool5 = L.Pooling(n.relu5, name='pool5', kernel_size=3, stride=2, pool=P.Pooling.MAX)

    n.conv6 = L.Convolution(n.pool5, name='fc6-conv', kernel_size=6, num_output=4096,
                weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1),
                param=[dict(name='fc6_w', lr_mult=0.0, decay_mult=1), dict(name='fc6_b', lr_mult=0.0, decay_mult=0)])

    n.relu6 = L.ReLU(n.conv6, name='relu6', in_place=False)
    trailing = n.relu6
    if split == 'train':
        n.drop6 = L.Dropout(n.relu6, name='drop6', dropout_ratio=0.5, in_place=True)
        trailing = n.drop6

    n.conv7 = L.Convolution(trailing, name='fc7_-conv', kernel_size=1, num_output=4096,
                weight_filler=dict(type='gaussian', std=0.005), bias_filler=dict(type='constant', value=0.1),
                param=[dict(name='fc7__w', lr_mult=1, decay_mult=1), dict(name='fc7__b', lr_mult=1, decay_mult=1)])
    n.relu7 = L.ReLU(n.conv7, name='relu7', in_place=False)
    trailing = n.relu7
    if split == 'train':
        n.drop7 = L.Dropout(n.relu7, name='drop7', dropout_ratio=0.5, in_place=True)
        trailing = n.drop7

    n.conv8 = L.Convolution(trailing, name='fc8_output-conv', kernel_size=1, num_output=depth,
            weight_filler=dict(type='gaussian', std=0.01), bias_filler=dict(type='constant', value=0),
            param=[dict(name='fc8_output_w', lr_mult=1, decay_mult=1), dict(name='fc8_output_b', lr_mult=1, decay_mult=1)])
    return n


def BuildNet(sport, depth, split, version):
    n = BuildBaseNet(sport, depth, split)
    last = n.conv8
    # zu wenig parameter
    if version=='fixedfixedlearned':
        n.upsample1 = upsample('upsample1', last, depth, 4, 'upsample_weight')
        n.upsample2 = upsample('upsample2', n.upsample1, depth, 4, 'upsample_weight')
        n.deconv = deconv('deconv', n.upsample2, depth, 1, 0.01, 0)
        last = n.deconv
    # doesnt fit memory
    elif version=='fixedlearnedlearned':
        n.upsample1 = upsample('upsample1', last, depth, 4, 'upsample_weight')
        n.deconv1 = deconv('deconv1', n.upsample1, depth, 4, 0.01, 0)
        n.deconv2 = deconv('deconv2', n.deconv1, depth, 4, 0.01, 0)
        last = n.deconv2
    # sinnlos glaub ich
    elif version=='learnedmiddle':
        n.deconv1 = deconv('deconv1', last, depth, 4, 0.01, 0)
        n.deconv2 = deconv('deconv2', n.deconv1, depth, 4, 0.01, 0)
        n.deconv3 = deconv('deconv3', n.deconv2, depth, 1, 0.01, 0)
        last = n.deconv3
    elif version=='learnedsmall':
        n.deconv1 = deconv('deconv1', last, depth, 4, 0.01, 0)
        n.deconv2 = deconv('deconv2', n.deconv1, depth, 4, 0.01, 0)
        last = n.deconv2
    # doesnt fit memory
    elif version=='learnedbig':
        n.deconv1 = deconv('deconv1', last, depth, 4, 0.01, 0)
        n.deconv2 = deconv('deconv2', n.deconv1, depth, 4, 0.01, 0)
        n.deconv3 = deconv('deconv3', n.deconv2, depth, 4, 0.01, 0)
        last = n.deconv3
    elif version=='fixedlearned':
        n.upsample = upsample('upsample', last, depth, 2, 'upsample_weight')
        n.deconv1 = deconv('deconv', n.upsample, depth, 8, 0.01, 0)
        last = n.deconv1
    else:
        print 'NO KNOWN VERSION YOU FOUL'
        return

    if split == 'train' or split == 'test':
        n.crop = L.Crop(last, n.label, axis=2, offset=0)
        n.softmax = L.SoftmaxWithLoss(n.crop, n.label, loss_param=dict(ignore_label=-1, normalize=False))
    if split == 'deploy':
        #n.crop = L.Crop(last, n.data, axis=2, offset=0)
        pass
    return n.to_proto()


def main(argv):
    sport = 'long_jump'
    depth = 468
    ver = 'learnedsmall'
    base = './fcn/'+sport+'/'
    proto = BuildNet(sport,depth,'train',ver)
    with open(base+'train.prototxt', 'w') as f:
        f.write(str(proto))
    proto = BuildNet(sport,depth,'test',ver)
    with open(base+'test.prototxt', 'w') as f:
        f.write(str(proto))
    proto = BuildNet(sport,depth,'deploy',ver)
    with open(base+'deploy.prototxt', 'w') as f:
        f.write(str(proto))


if __name__ == "__main__":
    main(sys.argv[1:])
