#!/usr/bin/env python
# -*- coding: utf-8 -*-


import six
import time
import numpy as np
import random
np.random.seed(1234)
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
        dataset = self.make_random_dataset(arg.datasize, args.seq_length, args.n_input, 'OFF', args.random_length)
        dataset_test = self.make_random_dataset(arg.datasize, args.seq_length, args.n_input, 'ON', args.random_length)
        self.dataset = self.make_minibatch(dataset, args.batchsize)
        self.dataset_test = self.make_minibatch(dataset_test, args.batchsize)
    
    def make_random_dataset(self, datasize=10000, seq_length=20, n_input=100, volatile='OFF', random_length=False):
        if random_length:
            # Todo: ここを最大長を指定してランダムな長さにする
            dataset = [np.random.normal(0.0, 1.0, (random.randint(1, seq_length), n_input)) for _ in xrange(datasize)]
        else:
            dataset = np.random.normal(0.0, 1.0, (datasize, seq_length, n_input))
            dataset = dataset.tolist()
        dataset = [Variable(self.xp.array(d, dtype=self.xp.float32), volatile=volatile) for d in dataset]
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
    dataset_test = test_obj.dataset_test

    nn = net.NStepLSTM(n_layer=args.n_layer, n_vocab=args.n_vocab, n_input=args.n_input, n_units=args.n_units, dropout=args.dropout, cudnn=args.cudnn)
    opt = optimizers.SGD()
    opt.setup(nn)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        nn.to_gpu()

    avg_time_forward = []
    avg_time_backward = []
    avg_time_forward_test = []

    # n epoch
    for i in xrange(args.n_epoch):
        sum_forward_time = 0.0
        sum_backward_time = 0.0
        for input_data in dataset:
            hx = chainer.Variable(
                    xp.zeros((args.n_layer, len(input_data), args.n_units), dtype=xp.float32), volatile="auto")
            cx = chainer.Variable(
                    xp.zeros((args.n_layer, len(input_data), args.n_units), dtype=xp.float32), volatile="auto")
            if args.gpu >= 0:
                chainer.cuda.to_cpu(cx.data)
            # forward
            start_time = test_obj.start_time()
            ys = nn(hx, cx, input_data)
            if args.gpu >= 0:
                chainer.cuda.to_cpu(ys[0].data)
            end_time = test_obj.end_time()
            time_forward = end_time - start_time
            
            # loss
            loss = sum([F.sum(_ys) for _ys in ys])
            # update
            nn.zerograds()
            start_time = test_obj.start_time()
            # backward
            loss.backward()
            if args.gpu >= 0:
                chainer.cuda.to_cpu(loss.data)
            end_time = test_obj.end_time()
            time_backward = end_time - start_time
            opt.update()

            sum_forward_time += time_forward
            sum_backward_time += time_backward
            # print "time_forward :", time_forward
            # print "time_backward:", time_backward

        sum_forward_time_test = 0.0
        # test data
        for input_data in dataset_test:
            hx = chainer.Variable(
                    xp.zeros((args.n_layer, len(input_data), args.n_units), dtype=xp.float32), volatile="auto")
            cx = chainer.Variable(
                    xp.zeros((args.n_layer, len(input_data), args.n_units), dtype=xp.float32), volatile="auto")
            if args.gpu >= 0:
                chainer.cuda.to_cpu(cx.data)
            # forward
            start_time = test_obj.start_time()
            ys = nn(hx, cx, input_data)
            if args.gpu >= 0:
                chainer.cuda.to_cpu(ys[0].data)
            end_time = test_obj.end_time()
            time_forward = end_time - start_time

            sum_forward_time_test += time_forward
            # print "time_forward (test) :", time_forward
            # print ys[0].data.shape

        avg_time_forward.append(sum_forward_time)
        avg_time_forward_test.append(sum_forward_time_test)
        avg_time_backward.append(sum_backward_time)
        print i, " time_forward       :", sum_forward_time
        print i, " time_forward (test):", sum_forward_time_test
        print i, " time_backward      :", sum_backward_time
        print '------------------------'

    print "avg_time_forward:", float(sum(avg_time_forward)) / len(avg_time_forward)
    print "avg_time_forward_test:", float(sum(avg_time_forward_test)) / len(avg_time_forward_test)
    print "avg_time_backward:", float(sum(avg_time_backward)) / len(avg_time_backward)



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
    parser.add_argument('--random_length', dest='random_length', type=int, default=0, help='random_length')
    parser.add_argument('--datasize', type=int, dest='datasize', default=10000, help='datasize')
    parser.add_argument('--cudnn', default=1, type=int, help='cudnn')


    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=50, help='n_epoch')
    parser.add_argument('--save_model', dest='save_model', type=int, default=0, help='n_epoch')

    args = parser.parse_args()
    print args
    test_performance(args)