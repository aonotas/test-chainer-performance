#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os.path

def read(filename):
    times = []
    f = open(filename)
    # build parameters
    namespace = f.readline().strip()
    namespace = namespace.replace("Namespace(", "").replace(")", "")
    namespaces = namespace.split(", ")
    namespace_dict = dict([(d.split("=")[0], "".join(d.split("=")[1:])) for d in namespaces])
    time_forward = []
    time_forward_test = []
    time_backward = []
    for l in f:
        l = l.strip()
        if "avg_" in l:
            name = l.split(":")[0].strip()
            avg_time = float(l.split(":")[-1])

            if name == "avg_time_forward":
                avg_time_forward = avg_time
            elif name == "avg_time_backward":
                avg_time_backward = avg_time
            else:
                avg_time_forward_test = avg_time
        elif ":" in l:
            # iteration = int(l.split(":")[0][0])
            name = l.split(":")[0][1:].strip()
            time = float(l.split(":")[-1])
            if name == "time_forward":
                time_forward.append(time)
            elif name == "time_backward":
                time_backward.append(time)
            else:
                time_forward_test.append(time)
        elif "--------" in l:
            pass
    
    avg_times = [avg_time_forward, avg_time_forward_test, avg_time_backward]
    times = [time_forward, time_forward_test, time_backward]
    return namespace_dict, avg_times, times

def is_file_exist(filename):
    return os.path.isfile(filename)

def get_filesize(path):
    return os.path.getsize(path)

def show(filename):
    namespace_dict, avg_times, times = read(filename=filename)
    params = ["batchsize", "seq_length", "random_length", "n_layer", "n_input", "n_units", "dropout"]
    values = [namespace_dict[param] for param in params] + [str(l) for l in avg_times]

    filename_cudnn = filename.replace('cudnn-0', 'cudnn-1')
    if not is_file_exist(filename_cudnn):
        return None
    if get_filesize(filename_cudnn) < 1000:
        return None
    # CUDNN
    namespace_dict, avg_times_cudnn, times_cudnn = read(filename=filename_cudnn)
    values += [str(l) for l in avg_times_cudnn]

    # 倍率
    values += map(str, [avg_times[i]/avg_times_cudnn[i] for i in xrange(3)])

    # 総合時間
    values += map(str, [sum(times[i]) for i in xrange(3)])
    values += map(str, [sum(times_cudnn[i]) for i in xrange(3)])

    print "\t".join(values)



if __name__ == '__main__':
    # filename = "./log/log_datasize-10000_n_input-128_n_units-512_batchsize-128_seq_length-5_cudnn-1_n_layer-2_gpu-0_n_epoch-10.txt"
    
    # import glob
    # log_filenames = glob.glob("./log/*.txt")
    # # print log_filenames
    # for filename in log_filenames:
    #     show(filename)
    import sys
    filename = sys.argv[1]
    for l in open(filename):
        l = l.strip()
        if "cudnn-0" in l:
            show(l)
