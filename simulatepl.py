from theseusqfls import genplanck
import numpy as np
import h5py
import matplotlib.pyplot as plt
n_data = 100
n_pixels = 1024


energy = np.linspace(1.4, 2.0, n_data)
qfls = 1.1*np.ones((n_pixels, n_pixels)) + 0.05*1.1*np.random.random((n_pixels, n_pixels))
gamma = 0.01*np.ones((n_pixels, n_pixels)) + 0.05*0.01*np.random.random((n_pixels, n_pixels))
theta = 1.25*np.ones((n_pixels, n_pixels)) + 0.05*1.25*np.random.random((n_pixels, n_pixels))
bandgap = 1.65*np.ones((n_pixels, n_pixels)) + 0.05*1.65*np.random.random((n_pixels, n_pixels))

pl = np.zeros((n_data, n_pixels, n_pixels))

for i in range(n_pixels):
    for j in range(n_pixels):
        pl[:, i, j] = genplanck.gen_planck(energy, qfls[i, j], gamma[i, j], theta[i, j], bandgap[i, j])

fig, ax = plt.subplots()
ax.plot(energy, np.mean(pl, axis=(1, 2)))
plt.show()

with h5py.File('theseusqfls/data/example_photometric.h5', 'w') as f:
    f['energy'] = energy
    f['pl'] = pl
    f['qfls'] = qfls
    f['gamma'] = gamma
    f['theta'] = theta
    f['bandgap'] = bandgap