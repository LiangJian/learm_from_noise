from error_and_resample import *
import numpy as np


def get_distribution(data0_, n_sample0_, nboot_=50):
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

    distribution_boot = []
    for ib in range(nboot_):
        data_boot_ = np.sort(get_boot_sample(data0_))
        distribution_tmp = np.zeros(n_sample)
        for ele in data_boot_:
            i = int((ele - st) // interval)
            distribution_tmp[i] += 1
        distribution_boot.append(distribution_tmp)
    distribution_boot = np.array(distribution_boot)
    distribution_boot = distribution_boot.reshape(nboot_, int(distribution_boot.size/nboot_))
    distribution_err = np.std(distribution_boot, 0)

    return vals[:n_sample], distribution[:n_sample]/data0_.size, distribution_err/data0_.size

