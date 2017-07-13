from error_and_resample import *
import numpy as np


def get_distribution(data0_, n_sample0_):
    data_ = np.sort(data0_)
    st = data_[0]
    ed = data_[-1]

    n_sample = n_sample0_ + 1
    interval = (ed - st) / n_sample0_
    distribution = np.zeros(n_sample)
    for ele in data_:
        i = int((ele - st) // interval)
        distribution[i] += 1
    vals = np.arange(st, ed + interval, interval)
    return vals, distribution
