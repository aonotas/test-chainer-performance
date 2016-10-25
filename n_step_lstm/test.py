#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


def make_random_dataset(args):
    pass

def test_performance(args):
    pass


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', '-g', default=-1, type=int, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--batchsize', dest='batchsize', type=int, default=20, help='learning minibatch size')
    parser.add_argument('--window', type=int, dest='window', default=5, help='window')
    parser.add_argument('--hidden_size', dest='hidden_size', default=100, type=int, help='number of units')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0.0, help='weight_decay')
    parser.add_argument('--n_epoch', dest='n_epoch', type=int, default=50, help='n_epoch')
    parser.add_argument('--save_model', dest='save_model', type=int, default=0, help='n_epoch')

    args = parser.parse_args()

    test_performance(args)