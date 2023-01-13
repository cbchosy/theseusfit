import torch
import theseus as th
import matplotlib.pyplot as plt
import h5py
import math
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline

LOOKUP_TABLE = 'data/lookup_table.csv'
h = 4.1357e-15
c = 2.9979e8
k = 8.6173e-5
q = 1.602e-19
T = 298.15
alpha = 10

nx = round(1.5 / 0.05) + 1
ny = round((100 + 60) / 0.1) + 1
d = pd.read_csv(LOOKUP_TABLE, names=['G'], usecols=[2])
theta = np.linspace(0.5, 2, nx)
theta_fine = np.linspace(0.5, 2, nx*50)
energy = np.linspace(-60, 100, ny)
energy_fine = np.linspace(-60, 100, ny*50)
d = np.reshape(d['G'].to_numpy(), (nx, ny))
G = torch.tensor(RectBivariateSpline(theta, energy, d, kx=3, ky=3, s=0)(theta_fine, energy_fine))

torch.manual_seed(0)


def import_hyperspectral(filename):
    with h5py.File(filename, 'r') as f:
        wavelength = np.array(f['Cube/Wavelength'], dtype=np.float64)
        cube = np.array(f['Cube/Images'], dtype=np.float64)
    energy = 1E9 * h * c / np.flip(wavelength, axis=0)
    cube = np.clip(cube, 0, None)
    cube = np.flip(cube, axis=0)
    cube = 10000 * math.pi * cube
    cube = cube[:, 100:104, 100:104]
    cube = cube.reshape((cube.shape[0], -1)).T
    energy = np.stack([energy]*cube.shape[0], axis=0)
    return energy, cube


def interp_index(arr, val):
    arr_min = val.unsqueeze(-1) - arr.unsqueeze(0)
    values, indices = torch.min(torch.abs(arr_min), -1)
    return indices


def generalized_planck_error(optim_vars, aux_vars):
    qfls, gamma, theta, bandgap = optim_vars
    E, I = aux_vars
    i_theta = interp_index(theta_fine, theta.tensor)
    i_energy = interp_index(energy_fine, (E.tensor - bandgap.tensor)/gamma.tensor)
    part1 = 2 * math.pi * E.tensor ** 2 / (h ** 3 * c ** 2)
    part2 = 1 / (math.e ** ((E.tensor - qfls.tensor) / (k * T)) - 1)
    part3 = (1 - math.e ** (-40 * torch.sqrt(gamma.tensor) * G[i_theta, i_energy]))
    part4 = (1 - 2 / (math.e ** ((E.tensor - qfls.tensor) / (2 * k * T)) + 1))
    err = I.tensor - part1 * part2 * part3 * part4
    return err

def generate_gen_planck(E, qfls, gamma, theta, bandgap):
    i_theta = interp_index(theta_fine, theta)
    i_energy = interp_index(energy_fine, (E - bandgap.unsqueeze(-1))/gamma.unsqueeze(-1))
    part1 = 2 * math.pi * E ** 2 / (h ** 3 * c ** 2)
    part2 = 1 / (math.e ** ((E - qfls.unsqueeze(-1)) / (k * T)) - 1)
    part3 = (1 - math.e ** (-40 * torch.sqrt(gamma.unsqueeze(-1)) * G[i_theta.unsqueeze(-1), i_energy]))
    part4 = (1 - 2 / (math.e ** ((E - qfls.unsqueeze(-1)) / (2 * k * T)) + 1))
    return torch.squeeze(part1 * part2 * part3 * part4)



# def generalized_planck_error(optim_vars, aux_vars):
#     qfls, gamma, theta, bandgap = optim_vars
#     E, I = aux_vars
#     err = I.tensor - qfls.tensor*E.tensor
#     return err


filename = r'/Volumes/GoogleDrive/Shared drives/StranksLab/Personal Folders/Cullen Chosy/Hyperspectral/20221102-Surrey/MF1/20x_region1_photometric.h5'

E_data, I_data = import_hyperspectral(filename)

E = th.Variable(torch.tensor(E_data), name='E')
I = th.Variable(torch.tensor(I_data), name='I')
theta_fine = torch.tensor(theta_fine)
energy_fine = torch.tensor(energy_fine)

qfls_data = 1.15*torch.ones(I_data.shape[0], 1, dtype=torch.float64)
gamma_data = 0.05*torch.ones(I_data.shape[0], 1, dtype=torch.float64)
theta_data = 1.5*torch.ones(I_data.shape[0], 1, dtype=torch.float64)
bandgap_data = 1.65*torch.ones(I_data.shape[0], 1, dtype=torch.float64)

# I_fit = generate_gen_planck(torch.tensor(E_data), qfls_data, gamma_data, theta_data, bandgap_data)
#
# fig, ax = plt.subplots()
# ax.plot(np.mean(np.array(E_data), axis=0), np.mean(np.array(I_fit), axis=0))
# ax.plot(np.mean(np.array(E_data), axis=0), np.mean(np.array(I_data), axis=0))
# plt.show()

qfls = th.Vector(1, name='qfls', dtype=torch.float64)
gamma = th.Vector(1, name='gamma', dtype=torch.float64)
theta = th.Vector(1, name='theta', dtype=torch.float64)
bandgap = th.Vector(1, name='bandgap', dtype=torch.float64)

optim_vars = qfls, gamma, theta, bandgap
aux_vars = E, I
cost_weight = th.ScaleCostWeight(scale=torch.tensor(1.0, dtype=torch.float64))
cost_function = th.AutoDiffCostFunction(
    optim_vars, generalized_planck_error, I_data.shape[-1], aux_vars=aux_vars, name='error_fn', cost_weight=cost_weight
)
objective = th.Objective(dtype=torch.float64)
objective.add(cost_function)
optimizer = th.GaussNewton(
    objective,
    max_iterations=50,
    step_size=0.1,
)
theseus_optim = th.TheseusLayer(optimizer)

theseus_inputs = {
    'E': torch.tensor(E_data),
    'I': torch.tensor(I_data),
    'qfls': qfls_data,
    'gamma': gamma_data,
    'theta': theta_data,
    'bandgap': bandgap_data
}
error = objective.error()
error_sq = objective.error_squared_norm()

fig, ax = plt.subplots()
ax.plot(np.array(E.tensor).T, np.array(error + I.tensor).T)
ax.plot(np.array(E.tensor).T, np.array(I.tensor).T)
plt.show()

with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={'track_best_solution': True,
                                          'verbose': True})
print("Best solution:", info.best_solution)










