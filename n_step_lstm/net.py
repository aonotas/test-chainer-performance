#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import chainer
import chainer.links as L

class NStepLSTM(chainer.Chain):

    def __init__(self, n_layer, n_vocab, n_input, n_units, dropout, cudnn=True):
        super(NStepLSTM, self).__init__(
            l1=L.NStepLSTM(n_layer, n_input, n_units,
                           dropout, use_cudnn=cudnn),
        )

    def __call__(self, hx, cx, input_x, train=True):
        hy, cy, ys = self.l1(hx, cx, input_x, train=train)
        return ys


def sample():
    # list
    xs = [xp.asarray(item, dtype=np.int32)
          for item in train_now[perm[i:i + args.batchsize]]]


    hx = chainer.Variable(
        xp.zeros((args.layer, len(xs), args.unit), dtype=xp.float32))
    cx = chainer.Variable(
        xp.zeros((args.layer, len(xs), args.unit), dtype=xp.float32))

    t = [xp.asarray(item, dtype=np.int32)
         for item in train_next[perm[i:i + args.batchsize]]]        