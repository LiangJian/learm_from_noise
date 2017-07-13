import matplotlib.pyplot as plt
from functions import get_distribution
import numpy as np

ip = 0
io = 0
it = 20
n_sample0 = 100

# load
data = np.load('two_p2_%d.npy' % ip)
nc = data.shape[0]
no = data.shape[1]
nt = data.shape[2]

# distribution of the imaginary part
data_pick = data[:, io, it]
data_imag = data_pick.imag
data_real = data_pick.real

x, y = get_distribution(data_imag, n_sample0)
plt.errorbar(x, y, yerr=np.sqrt(y), fmt='x')
plt.grid()
# plt.show()
plt.close()

x, y = get_distribution(data_real, n_sample0)
plt.errorbar(x, y, yerr=np.sqrt(y), fmt='x')
plt.grid()
# plt.show()
plt.close()

# correlation of the real and imaginary part
data_pick = data[:, io, :it]
data_imag = data_pick.imag
data_real = data_pick.real
data_real_ave = np.average(data_real, 0)
data_real_diff = np.array([data_real[:, i] - data_real_ave[i] for i in range(0, data_real.shape[1])]).swapaxes(0, 1)

corr = np.corrcoef(data_real, data_real, False)
print([corr[0, i] for i in range(20, 30)])
corr = np.corrcoef(data_real, data_imag, False)
print([corr[20, i] for i in range(20, 25)])
data_rand = np.random.random(data_real.size).reshape(data_real.shape)
corr = np.corrcoef(data_real, data_rand, False)
print([corr[0, i] for i in range(20, 25)])

A = np.random.random(100)*0.1 + 1
m1 = np.random.random(100)*0.01 + 0.1
m2 = np.random.random(100)*0.01 + 0.15
noise = (np.random.random(100 * 20)*0.1).reshape(100, 20)
data_fake = np.zeros(shape=(100, 20))
for ic in range(0, 100):
    for it in range(0, 20):
        data_fake[ic, it] = A[ic] * (np.exp(-m1[ic] * it) + np.exp(-m2[ic] * it)) + noise[ic, it]
corr = np.corrcoef(data_fake, rowvar=False)
print([corr[0, i] for i in range(0, 10)])
