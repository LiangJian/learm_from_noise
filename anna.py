import numpy as np

data = np.load('two_p2_0_bin.npy')
print(data.shape)
data_ave = np.average(data, 0)
data_err = np.std(data, 0)/np.sqrt(data.shape[0] - 1)
print(data_ave[0, :16].real)
print(data_ave[0, :16].imag)
print(data_err[0, :16])

