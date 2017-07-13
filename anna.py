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
plt.plot([float('%6.4f' % corr[0, i]) for i in range(20, 30)], 'x', label='real-real', markersize=12)
corr = np.corrcoef(np.abs(data_real), np.log(np.abs(data_real)), False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(20, 30)], '+', label='real-log(real)', markersize=12)
corr = np.corrcoef(np.log(np.abs(data_real)), np.log(np.abs(data_real)), False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(20, 30)], '*', label='log(real)-log(real)', markersize=12)
corr = np.corrcoef(data_real, data_imag, False)
plt.plot([float('%6.4f' % corr[20, i]) for i in range(20, 30)], 'x', label='real-imag', markersize=12)
data_rand = np.random.random(data_real.size).reshape(data_real.shape)
corr = np.corrcoef(data_real, data_rand, False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(20, 30)], 'x',label='real-rand', markersize=12)
plt.legend()
plt.grid()
plt.show()

nc = 100
nt = 50
noise = np.random.random(nc)
A1 = noise*0.1 + 1
A2 = noise*0.1 + 0.5
A3 = noise*0.1 + 10
m1 = noise*0.1 + 0.1
m2 = noise*0.5 + 0.5
m3 = noise*0.1 + 0.4
noise = (np.random.random(nc * nt)*0.1).reshape(nc, nt)
data_fake = np.zeros(shape=(nc, nt))
for it in range(0, nt):
    data_fake[..., it] = A1 * np.exp(-m1 * it) + A2 * np.exp(-m2 * it) + A3 * np.exp(-m3 * it)
corr = np.corrcoef(data_fake, rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], 'x', label='real-real', markersize=12)
corr = np.corrcoef(np.log(data_fake), rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], 'x', label='log(real)-log(real)', markersize=12)
plt.legend()
plt.grid()
plt.show()
