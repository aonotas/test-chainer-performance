#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.grid_search import ParameterGrid

"""
Example:
> python compare_script.py > command_list.txt
"""

params = {}
params["n_layer"] = [1, 2, 3]
params["batchsize"] = [127, 128, 200, 255, 256]
params["n_input"] = [128, 512]
params["n_units"] = [128, 512]
params["seq_length"] = [5, 25, 50]
params["random_length"] = [0, 1]
params["cudnn"] = [1, 0]
params["dropout"] = [0.0, 0.5]
params["datasize"] = [10000]
params["n_epoch"] = [10]
params["gpu"] = [0]



def main():
    # paterns = itertools.combinations(values, 1)
    paterns = ParameterGrid(params)
    for p in paterns:
        args = " ".join(["--"+k+"="+str(v) for k,v in p.items()])
        savename = "_".join([k+"-"+str(v) for k,v in p.items()])
        command = "python run.py "+ args + " > ./log/log_"+savename+".txt"
        # print args
        print command



if __name__ == '__main__':
    main()
