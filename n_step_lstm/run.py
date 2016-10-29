#!/usr/bin/env python
# -*- coding: utf-8 -*-

import six
import time
import numpy as np
import random
random.seed(1234)

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers
from chainer import Variable

import net

class TestPerformance(object):
    """TestPerformance"""
    def __init__(self, xp, arg):
        super(TestPerformance, self).__init__()
        self.arg = arg
        self.xp = xp
        dataset = self.make_random_dataset(arg.datasize, args.seq_length, args.n_input)
        self.dataset = self.make_minibatch(dataset, args.batchsize)
        
    def make_random_dataset(self, datasize=10000, seq_length=20, n_input=100):
        # Todo: ここを最大長を指定してランダムな長さにする
        dataset = np.random.normal(0.0, 1.0, (datasize, seq_length, n_input))
        print dataset
        dataset = [Variable(self.xp.array(d, dtype=self.xp.float32)) for d in dataset.tolist()]
        return dataset

    def make_minibatch(self, dataset, batchsize):
        n_dataset = len(dataset)
        dataset_batch = []
        for i in six.moves.range(0, n_dataset, batchsize):
            input_data = dataset[i:i + batchsize]
            dataset_batch.append(input_data)
        return dataset_batch

    def start_time(self):
        self.start = time.time()
        return self.start

    def end_time(self):
        self.end = time.time()
        return self.end


def test_performance(args):
    xp = cuda.cupy if args.gpu >= 0 else np
    test_obj = TestPerformance(xp, args)
    dataset = test_obj.dataset

    nn = net.NStepLSTM(n_layer=args.n_layer, n_vocab=args.n_vocab, n_input=args.n_input, n_units=args.n_units, dropout=args.dropout)
    opt = optimizers.SGD()
    opt.setup(nn)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        nn.to_gpu()

    for i in xrange(args.n_epoch):
        start_time = test_obj.start_time()
        for input_data in dataset:
            hx = chainer.Variable(xp.zeros((args.n_layer, len(input_data), args.n_units), dtype=xp.float32))
            cx = chainer.Variable(xp.zeros((args.n_layer, len(input_data), args.n_units), dtype=xp.float32))
            ys = nn(hx, cx, input_data)
            loss = sum([F.sum(_ys, axis=0) for _ys in ys])
            nn.zerograds()
            loss.backward()
            opt.update()

            # print ys[0].data.shape
        end_time = test_obj.end_time()
        print end_time - start_time


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=20, help='learning minibatch size')
    parser.add_argument('--n_input', dest='n_input', type=int, default=100, help='n_input')
    parser.add_argument('--n_units', dest='n_units', type=int, default=200, help='n_units')
    parser.add_argument('--n_vocab', dest='n_vocab', type=int, default=10000, help='n_vocab')
    parser.add_argument('--n_layer', dest='n_layer', type=int, default=1, help='n_layer')
    parser.add_argument('--dropout', dest='dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--seq_length', type=int, dest='seq_length', default=5, help='seq_length')
    parser.add_argument('--datasize', type=int, dest='datasize', default=10000, help='seq_length')


    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=50, help='n_epoch')
    parser.add_argument('--save_model', dest='save_model', type=int, default=0, help='n_epoch')

    args = parser.parse_args()

    test_performance(args)