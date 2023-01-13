from typing import List, Optional, Tuple
import theseus as th
import torch
import math
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

torch.manual_seed(0)

LOOKUP_TABLE = 'data/lookup_table.csv'
h = 4.1357e-15
c = 2.9979e8
k = 8.6173e-5
q = 1.602e-19
T = 298.15
alpha = 10


def gen_lookup_table():
    nx = round(1.5 / 0.05) + 1
    ny = round((100 + 60) / 0.1) + 1
    d = pd.read_csv(LOOKUP_TABLE, names=['G'], usecols=[2])
    theta = np.linspace(0.5, 2, nx)
    theta_fine = np.linspace(0.5, 2, nx * 50, dtype=np.float64)
    energy = np.linspace(-60, 100, ny)
    energy_fine = np.linspace(-60, 100, ny * 50, dtype=np.float64)
    d = np.reshape(d['G'].to_numpy(), (nx, ny))
    G = RectBivariateSpline(theta, energy, d, kx=3, ky=3, s=0)
    d_theta = torch.tensor(G.partial_derivative(1, 0)(theta_fine, energy_fine), dtype=torch.float64)
    d_energy = torch.tensor(G.partial_derivative(0, 1)(theta_fine, energy_fine), dtype=torch.float64)
    G_tensor = torch.tensor(G(theta_fine, energy_fine), dtype=torch.float64)
    return torch.tensor(theta_fine), torch.tensor(energy_fine), d_theta, d_energy, G_tensor


def generate_gen_planck(E, qfls, gamma, theta, bandgap, noise_factor=0.0):
    theta_fine, energy_fine, d_theta, d_energy, G = gen_lookup_table()
    i_theta = interp_index(theta_fine, theta)
    i_energy = interp_index(energy_fine, (E - bandgap) / gamma)
    part1 = 2 * math.pi * E ** 2 / (h ** 3 * c ** 2) * math.e ** (-E / (k * T))
    part2 = math.e ** (qfls / (k * T))
    part3 = (1 - math.e ** (-40 * torch.sqrt(gamma) * G[i_theta, i_energy]))
    return part1 * part2 * part3 + torch.randn(part3.shape) * noise_factor


def interp_index(arr, val):
    arr_min = val.unsqueeze(-1) - arr.unsqueeze(0)
    values, indices = torch.min(torch.abs(arr_min), -1)
    return indices


class QFLSCost(th.CostFunction):
    def __init__(
            self,
            cost_weight: th.CostWeight,
            E: th.Variable,
            I: th.Variable,
            qfls: th.Vector,
            gamma: th.Vector,
            theta: th.Vector,
            bandgap: th.Vector,
            name: Optional[str] = None,
    ):
        super().__init__(cost_weight, name=name)
        self.E = E
        self.I = I
        self.qfls = qfls
        self.gamma = gamma
        self.theta = theta
        self.bandgap = bandgap

        self.theta_fine, self.energy_fine, self.d_theta, self.d_energy, self.G = gen_lookup_table()

        self.register_optim_vars(['qfls', 'gamma', 'theta', 'bandgap'])
        self.register_aux_vars(['E', 'I'])

    def gen_planck(self):
        i_theta = interp_index(self.theta_fine, self.theta.tensor)
        i_energy = interp_index(self.energy_fine, (self.E.tensor - self.bandgap.tensor)/self.gamma.tensor)
        part1 = 2 * math.pi * self.E.tensor ** 2 / (h ** 3 * c ** 2) * math.e ** (-self.E.tensor / (k * T))
        part2 = math.e ** (self.qfls.tensor / (k * T))
        part3 = (1 - math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]))
        return torch.squeeze(part1 * part2 * part3)

    def error(self) -> torch.Tensor:
        return self.gen_planck() - self.I.tensor

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        i_theta = interp_index(self.theta_fine, self.theta.tensor)
        i_energy = interp_index(self.energy_fine, (self.E.tensor - self.bandgap.tensor)/self.gamma.tensor)
        part1 = 2 * math.pi * self.E.tensor ** 2 / (h ** 3 * c ** 2) * math.e ** (-self.E.tensor / (k * T))
        part2 = math.e ** (self.qfls.tensor / (k * T))
        part3 = (1 - math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]))

        part3_dgamma = 40 * math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]) * (self.G[i_theta, i_energy] / (2 * torch.sqrt(self.gamma.tensor)) + (self.bandgap.tensor - self.E.tensor) / self.gamma.tensor ** (3/2) * self.d_energy[i_theta, i_energy])
        part3_dtheta = 40 * math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]) * torch.sqrt(self.gamma.tensor) * self.d_theta[i_theta, i_energy]
        part3_dbandgap = 40 * math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]) * (-1 / torch.sqrt(self.gamma.tensor)) * self.d_energy[i_theta, i_energy]

        j = [
            # jacobian of error function wrt qfls
            ((k * T) ** -1 * part1 * part2 * part3).unsqueeze(-1),
            # jacobian of error function wrt gamma
            (part1 * part2 * part3_dgamma).unsqueeze(-1),
            # jacobian of error function wrt theta
            (part1 * part2 * part3_dtheta).unsqueeze(-1),
            # jacobian of error function wrt bandgap
            (part1 * part2 * part3_dbandgap).unsqueeze(-1),
        ]
        return j, self.error()

    def dim(self) -> int:
        return self.I.tensor.shape[1]

    def _copy_impl(self, new_name: Optional[str] = None) -> "QFLSCost":
        return QFLSCost(
            self.weight.copy(), self.E.copy(), self.I.copy(), self.qfls.copy(), self.gamma.copy(), self.theta.copy(), self.bandgap.copy(), name=new_name
        )

batch_size = 10

E_data = torch.linspace(1.1, 1.8, 80, dtype=torch.float64).repeat(batch_size, 1)

qfls_data = 1.12*torch.ones(batch_size, 1, dtype=torch.float64)
gamma_data = 0.075*torch.ones(batch_size, 1, dtype=torch.float64)
theta_data = 1.55*torch.ones(batch_size, 1, dtype=torch.float64)
bandgap_data = 1.68*torch.ones(batch_size, 1, dtype=torch.float64)

qfls_init = 1.1*torch.ones(batch_size, 1, dtype=torch.float64)
gamma_init = 0.05*torch.ones(batch_size, 1, dtype=torch.float64)
theta_init = 1.5*torch.ones(batch_size, 1, dtype=torch.float64)
bandgap_init = 1.6*torch.ones(batch_size, 1, dtype=torch.float64)

qfls = th.Vector(1, name='qfls', dtype=torch.float64)
gamma = th.Vector(1, name='gamma', dtype=torch.float64)
theta = th.Vector(1, name='theta', dtype=torch.float64)
bandgap = th.Vector(1, name='bandgap', dtype=torch.float64)

I_data = generate_gen_planck(E_data, qfls_data, gamma_data, theta_data, bandgap_data, noise_factor=0.01)

E = th.Variable(E_data, name='E')
I = th.Variable(I_data, name='I')

cost_weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float64))

# construct cost functions and add to objective
objective = th.Objective(dtype=torch.float64)
cost_fn = QFLSCost(cost_weight, E, I, qfls, gamma, theta, bandgap)
objective.add(cost_fn)


#%%
error_sq = objective.error_squared_norm()

optimizer = th.LevenbergMarquardt(
    objective,
    max_iterations=100,
    step_size=0.1
)

theseus_optim = th.TheseusLayer(optimizer)

theseus_inputs = {
    'E': E_data,
    'I': I_data,
    'qfls': qfls_init,
    'gamma': gamma_init,
    'theta': theta_init,
    'bandgap': bandgap_init
}

objective.update(theseus_inputs)
#
# jac = cost_fn.jacobians()
# d_qfls = jac[0][0]/torch.abs(jac[0][0]).max()
# d_gamma = jac[0][1]/torch.abs(jac[0][1]).max()
# d_theta = jac[0][2]/torch.abs(jac[0][2]).max()
# d_bandgap = jac[0][3]/torch.abs(jac[0][3]).max()
#
# fig, ax = plt.subplots()
# ax.plot(np.array(I[0, :])/torch.abs(I[0, :]).max(), label='data')
# ax.plot(np.array(d_qfls)[0, :, 0], label='qfls')
# ax.plot(np.array(d_gamma)[0, :, 0], label='gamma')
# ax.plot(np.array(d_theta)[0, :, 0], label='theta')
# ax.plot(np.array(d_bandgap)[0, :, 0], label='bandgap')
# ax.legend()
# plt.show()
#%%
with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={'track_best_solution': True,
                                          'verbose': True})

print('Best solution:', info.best_solution)