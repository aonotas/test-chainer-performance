#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

def show(filename):
    namespace_dict, avg_times, times = read(filename=filename)
    params = ["cudnn", "batchsize", "seq_length", "n_layer", "n_input", "n_units"]
    values = [namespace_dict[param] for param in params] + [str(l) for l in avg_times]
    print "\t".join(values)



if __name__ == '__main__':
    # filename = "./log/log_datasize-10000_n_input-128_n_units-512_batchsize-128_seq_length-5_cudnn-1_n_layer-2_gpu-0_n_epoch-10.txt"
    
    import glob
    log_filenames = glob.glob("./log/*.txt")
    # print log_filenames
    for filename in log_filenames:
        show(filename)
