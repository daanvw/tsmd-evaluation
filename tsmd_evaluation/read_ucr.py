import os
import pandas as pd
import numpy as np

def load_ucr_dataset(path_to_data, ds_name, train=True):    
    train_or_test = "TRAIN" if train else "TEST"
    path_to_file = os.path.join(path_to_data, ds_name.lower(), f"{ds_name}_{train_or_test}.ts")
    time_series, labels = read_ts_file(path_to_file)  

    lengths = [len(series) for series in time_series]
    df = pd.DataFrame({"series": time_series, "label": labels, "length": lengths})
    return df

def read_ts_file(path_to_file):
    f = open(path_to_file)
    line = ""

    instances = []
    labels = []
    
    ndim = 1

    while not line.startswith("@data"):
        line = next(f)
        if line.startswith("@dimensions"):
            ndim = int(line.split()[1])

    lines = [line.replace(':', ',').split(',') for line in f.readlines()]
    for line in lines:
        l = len(line) - 1
        instance = np.array(line[:l], dtype=float)

        if (l % ndim == 0):
            instance = instance.reshape(l // ndim, ndim, order='F')
            instances.append(instance)
            labels.append(line[-1].strip())

    return instances, labels