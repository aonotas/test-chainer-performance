#!/usr/bin/env python
# -*- coding: utf-8 -*-


import chainer
import numpy as np
# 長さ順にソートしておく
x1 = chainer.Variable(np.array([0, 1, 2, 3, 4], dtype=np.int32))
x2 = chainer.Variable(np.array([4, 5, 6], dtype=np.int32))
x3 = chainer.Variable(np.array([4, 5], dtype=np.int32))

x_data = [x1, x2, x3]
batchsize = len(x_data)
x_dataset = chainer.functions.transpose_sequence(x_data)
# Auto-encoderの場合
y_data = x_data[:]
y_dataset = chainer.functions.transpose_sequence(y_data)

vocab_size = 2000
n_units = 200
embedding_size = 200
embID = chainer.links.EmbedID(vocab_size, embedding_size)
embID_decoder = chainer.links.EmbedID(vocab_size, embedding_size)

# lstm = chainer.links.LSTM(in_size=10, out_size=10)
encoder_lstm = chainer.links.StatelessLSTM(in_size=embedding_size, out_size=n_units)
decoder_lstm = chainer.links.StatelessLSTM(in_size=embedding_size, out_size=n_units)

output_layer = chainer.links.Linear(n_units, vocab_size)

x_len = len(x_dataset[0])
# c, h は初期化するべき
c = chainer.Variable(np.zeros((x_len, n_units), dtype=np.float32))
h = chainer.Variable(np.zeros((x_len, n_units), dtype=np.float32))

h_list = []
for i, x in enumerate(x_dataset):
    print "-" * 10
    x = embID(x)
    x_len = x.data.shape[0]
    h_len = h.data.shape[0]
    print "x_len:", x_len
    print "h_len:", h_len
    if x_len < h_len:
        h, h_stop = chainer.functions.split_axis(h, [x_len], axis=0)
        c, c_stop = chainer.functions.split_axis(c, [x_len], axis=0)
        # 処理済みのhをリストに追加
        h_list.append(h_stop)
        print "h:", h.data.shape
        print "c:", c.data.shape

    c, h = encoder_lstm(c, h, x)
    # print h.data


h_list.append(h)
# appendの順番的にリバースしておいた方が自然？
h_list.reverse()


h_encoded = chainer.functions.concat(h_list, axis=0)
print h_encoded.data.shape
# print h_encoded.data



def _make_tag(_batchsize, tag=0):
    shape = (_batchsize,)
    return np.full(shape, tag, dtype=np.int32)



x_len = len(x_dataset[0])
c = chainer.Variable(np.zeros((x_len, n_units), dtype=np.float32))
# h = chainer.Variable(np.zeros((x_len, out_size), dtype=np.float32))
h = h_encoded
start_tag = _make_tag(batchsize, tag=0)
start_tag = [chainer.Variable(start_tag)]

end_tag = _make_tag(batchsize, tag=1)
end_tag = [chainer.Variable(end_tag)]
# y = start_tag


decode_start_idx = 0
# decode

# y_datasetは<s>で始まる前提にする？
# ミニバッチ化する時に<eos>の扱いが面倒なので、データの前処理のときに
# [0, 1, 2, 3, <eos>]
# [0, 3, <eos>]
# [0, 1, 2, <eos>]
# とするほうが良さげ

y_dataset = list(y_dataset)
# for target in y_dataset:
for y, t in zip(start_tag + y_dataset[:-1], y_dataset[1:]):
    print "-" * 10
    y_embedding = embID(y)
    # y_len = y_embedding.data.shape[0]
    y_len = y_embedding.data.shape[0]
    # t_len = t.data.shape[0]
    h_len = h.data.shape[0]
    target_len = t.data.shape[0]
    # print t
    # print t_len
    print "y_len:", y_len
    print "target_len:", target_len
    if target_len < h_len:
        h, h_stop = chainer.functions.split_axis(h, [target_len], axis=0)
        c, c_stop = chainer.functions.split_axis(c, [target_len], axis=0)

    if target_len < y_len:
        y_embedding, _stop_y_embedding = chainer.functions.split_axis(y_embedding, [target_len], axis=0)

    print "y_embedding:", y_embedding.data.shape
    print "h:", h.data.shape
    c, h = encoder_lstm(c, h, y_embedding)

    predict = output_layer(h)
    print "predict:", predict.data.shape
    print h
    # x_len = x.data.shape[0]
    # h_len = h.data.shape[0]
    # embID_decoder()
    # loss = functions.softmax_cross_entropy(y, t)
    # x = embID(x)
    # x_len = x.data.shape[0]
    # h_len = h.data.shape[0]

