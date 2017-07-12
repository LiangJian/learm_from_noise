import numpy as np
import matplotlib.pyplot as plt

ip = 0
io = 0
it = 0

data = np.load('two_p2_%d.npy' % ip)
nc = data.shape[0]
no = data.shape[1]
nt = data.shape[2]
data_ave = np.average(data, 0)
data_err = np.std(data, 0)/np.sqrt(nc - 1)

data = data[:, io, it]

data_imag = data[...].imag
data_imag = np.sort(data_imag)
st = data_imag[0]
ed = data_imag[-1]
n_sample0 = 100
n_sample = n_sample0 + 1
interval = (ed - st) / n_sample0
distribution = np.zeros(n_sample)
for ele in data_imag:
    i = int((ele-st)//interval)
    distribution[i] += 1
plt.errorbar(np.arange(st, ed + interval, interval), distribution, yerr=np.sqrt(distribution), fmt='x')
plt.grid()
plt.show()
plt.close()
