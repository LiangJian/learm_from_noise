from fitting import *
import matplotlib.pyplot as plt
from functions import get_distribution
import numpy as np
import subprocess
from matplotlib.backends.backend_pdf import PdfPages
import lsqfit
import gvar as gv

pdf = PdfPages('anna.pdf')

ip = 0
io = 0
n_sample0 = 40

# load
data = np.load('two_p2_%d.npy' % ip)
nc = data.shape[0]
no = data.shape[1]
nt = data.shape[2]
print(nc, no, nt)
nc = 5000

#############################   O _ O   ###################################

data_pick = data[:nc, io, :]
data_jack = do_jack(data_pick, 0)
data_jack_ave = np.average(data_jack, 0)
data_jack_err = get_jack_error(data_jack, 0)
plt.yscale('log')
plt.errorbar(np.arange(30, 48, 1), -data_jack_ave[30:48].real, yerr=data_jack_err[30:48].real, fmt='x', label='C2')
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

#############################   O _ O   ###################################

plt.title('dis of imag')
for it in range(0, 4, 1):
    data_pick = data[:nc, io, it]
    data_imag = data_pick.imag
    x, y, z = get_distribution(data_imag, n_sample0)
    plt.errorbar(x, y, yerr=z, fmt='x', label='t=%02d' % it)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

plt.title('dis of real')
for it in range(0, 4, 1):
    data_pick = data[:nc, io, it]
    data_real = data_pick.real
    x, y, z = get_distribution(data_real, n_sample0)
    plt.errorbar(x-x[int(x.size/2)], y, yerr=z, fmt='x', label='t=%02d' % it)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()


def normal(x_, p_):
    A_ = p_['A']
    nu_ = p_['nu']
    sigma_ = p_['sigma']
    return A_ * np.exp(-(x_ - nu_)**2/(2 * sigma_**2))


it = 10
cut1 = 5
cut2 = 5

data_pick = data[:nc, io, it]
x, y, z = get_distribution(data_pick.imag, n_sample0)
x_cut = x[cut1:x.size - cut2]
y_cut = y[cut1:y.size - cut2]
z_cut = z[cut1:z.size - cut2]
x_cut = x_cut/np.average(x_cut)
p0 = {}
p0['A'] = np.average(y[int(y.size/2)])
p0['nu'] = np.average(x_cut)
p0['sigma'] = np.std(x_cut)
yy_cut = gv.gvar(y_cut, z_cut)
fit = lsqfit.nonlinear_fit(data=(x_cut, yy_cut), p0=p0, fcn=normal, debug=True)
print(fit.p0)
print_fit(fit)
x = x/np.average(x)
plt.title('fit the dis of imag')
plt.errorbar(x, y, yerr=z, fmt='x', label='t=%02d' % it)
plt.plot(x, normal(x, {'A':fit.p['A'].mean, 'nu':fit.p['nu'].mean, 'sigma':fit.p['sigma'].mean}))
plt.legend()
plt.grid()
pdf.savefig()
plt.close()


#############################   O _ O   ###################################

it = 50

# correlation of the real and imaginary part
data_pick = data[:nc, io, :it]
data_real = data_pick.real
data_imag = data_pick.imag

corr = np.corrcoef(data_real, data_real, False)
plt.plot([float('%6.4f' % corr[48, i]) for i in range(it+40, it + 50)], 'x', label='real-real', markersize=12)

corr = np.corrcoef((-data_real), np.log(-(data_real)), False)
plt.plot([float('%6.4f' % corr[48, i]) for i in range(it+40, it + 50)], '+', label='real-log(real)', markersize=12)

corr = np.corrcoef(np.log(-(data_real)), np.log(-(data_real)), False)
plt.plot([float('%6.4f' % corr[48, i]) for i in range(it+40, it + 50)], '*', label='log(real)-log(real)', markersize=12)

corr = np.corrcoef(data_real, data_imag, False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], 'x', label='real-imag', markersize=12)

corr = np.corrcoef(data_imag, data_imag, False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], '+', label='imag-imag', markersize=12)

data_rand = np.random.random(data_real.size).reshape(data_real.shape)
corr = np.corrcoef(data_real, data_rand, False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], 'x', label='real-rand', markersize=12)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()


data_real_ave = np.average(data_real, 0)
data_imag_ave = np.average(data_imag, 0)
data_real_std = np.std(data_real, 0)
data_imag_std = np.std(data_imag, 0)
data_real_err = data_real_std / np.sqrt(data_real.shape[0] - 1)
data_imag_err = data_imag_std / np.sqrt(data_imag.shape[0] - 1)
data_makeup = -data_imag * (data_real_err / data_imag_ave).reshape(1, it) * 0.4


corr = np.corrcoef(np.log(np.abs(data_real + data_makeup)), np.log(np.abs(data_real + data_makeup)), False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], '+', label='real(m)-real(m)', markersize=12)
corr = np.corrcoef(np.log(np.abs(data_real)), np.log(np.abs(data_real)), False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], 'x', label='real-real', markersize=12)
corr = np.corrcoef(np.abs(data_imag), np.abs(data_imag), False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], 'x', label='imag-imag', markersize=12)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

data_pick = do_jack(data_pick, 0)
data_real = data_pick.real
data_imag = data_pick.imag

corr = np.corrcoef(data_real, data_real, False)
plt.plot([float('%6.4f' % corr[20, i]) for i in range(it+20, it + 50)], 'x', label='real-real', markersize=12)

corr = np.corrcoef((-data_real), np.log(-(data_real)), False)
plt.plot([float('%6.4f' % corr[20, i]) for i in range(it+20, it + 50)], '+', label='real-log(real)', markersize=12)

corr = np.corrcoef(np.log(-(data_real)), np.log(-(data_real)), False)
plt.plot([float('%6.4f' % corr[20, i]) for i in range(it+20, it + 50)], '*', label='log(real)-log(real)', markersize=12)

corr = np.corrcoef(data_real, data_imag, False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], 'x', label='real-imag', markersize=12)

corr = np.corrcoef(data_imag, data_imag, False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], '+', label='imag-imag', markersize=12)

data_rand = np.random.random(data_real.size).reshape(data_real.shape)
corr = np.corrcoef(data_real, data_rand, False)
plt.plot([float('%6.4f' % corr[0, i]) for i in range(it, it + 30)], 'x', label='real-rand', markersize=12)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()


nc = 100
nt = 10
noise = np.random.normal(size=nc)
# noise = np.exp(noise)
# noise = np.random.random(size=nc)
A1 = noise*0.1 + 1.0
A2 = noise*0.1 + 1.3
A3 = noise*0.1 + 1.6
noise = np.exp(noise)
m1 = noise*0.1 + 0.1
m2 = noise*0.1 + 0.2
m3 = noise*0.1 + 0.3
noise = (np.random.random(nc * nt)*0.1).reshape(nc, nt)
data_fake = np.zeros(shape=(nc, nt))
for it in range(0, nt):
    data_fake[..., it] = A1 * np.exp(-m1 * it) + A2 * np.exp(-m2 * it) + A3 * np.exp(-m3 * it)
data_fake1 = np.copy(data_fake)
corr = np.corrcoef(data_fake, rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], 'x', label='real-real', markersize=12)
corr = np.corrcoef(np.log(np.abs(data_fake)), rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], '+', label='log(real)-log(real)', markersize=12)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

data_fake = do_jack(data_fake, 0)
corr = np.corrcoef(data_fake, rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], 'x', label='real-real', markersize=12)
corr = np.corrcoef(np.log(np.abs(data_fake)), rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], '+', label='log(real)-log(real)', markersize=12)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()

data_fake = do_anti_jack(data_fake, 0)
print(np.sum(data_fake - data_fake1))
corr = np.corrcoef(data_fake, rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], 'x', label='real-real', markersize=12)
corr = np.corrcoef(np.log(np.abs(data_fake)), rowvar=False)
plt.plot([float('%6.4f' % corr[1, i]) for i in range(1, nt)], '+', label='log(real)-log(real)', markersize=12)
plt.legend()
plt.grid()
pdf.savefig()
plt.close()


pdf.close()

#############################   O _ O   ###################################

subprocess.call(['open', 'anna.pdf'])