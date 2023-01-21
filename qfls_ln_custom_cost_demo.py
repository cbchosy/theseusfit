from typing import List, Optional, Tuple
import theseus as th
import torch
import math
import numpy as np
import pandas as pd
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt

torch.manual_seed(0)

LOOKUP_TABLE = 'theseusqfls/data/lookup_table.csv'
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
    theta_fine = np.linspace(0.5, 2, nx * 50)
    energy = np.linspace(-60, 100, ny)
    energy_fine = np.linspace(-60, 100, ny * 50)
    d = np.reshape(d['G'].to_numpy(), (nx, ny))
    G = RectBivariateSpline(theta, energy, d, kx=3, ky=3, s=0)
    d_theta = torch.tensor(G.partial_derivative(1, 0)(theta_fine, energy_fine), dtype=torch.float32)
    d_energy = torch.tensor(G.partial_derivative(0, 1)(theta_fine, energy_fine), dtype=torch.float32)
    G_tensor = torch.tensor(G(theta_fine, energy_fine), dtype=torch.float32)
    return torch.tensor(theta_fine, dtype=torch.float32), torch.tensor(energy_fine, dtype=torch.float32), d_theta, d_energy, G_tensor


def generate_gen_planck(E, qfls, gamma, theta, bandgap):
    theta_fine, energy_fine, d_theta, d_energy, G = gen_lookup_table()
    i_theta = interp_index(theta_fine, theta)
    i_energy = interp_index(energy_fine, (E - bandgap) / gamma)
    part1 = 2 * math.pi * E ** 2 / (h ** 3 * c ** 2) * math.e ** (-E / (k * T))
    part2 = math.e ** (qfls / (k * T))
    part3 = 1 - math.e ** (-40 * torch.sqrt(gamma) * G[i_theta, i_energy])
    return part1 * part2 * part3


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
        return torch.squeeze(
            # Part 1
            torch.log(2 * math.pi * self.E.tensor ** 2 / (h ** 3 * c ** 2)) - self.E.tensor / (k * T) +
            # Part 2
            self.qfls.tensor / (k * T) +
            # Part 3
            torch.log(1 - math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]))
        )

    def error(self) -> torch.Tensor:
        return self.gen_planck() - torch.log(self.I.tensor)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        i_theta = interp_index(self.theta_fine, self.theta.tensor)
        i_energy = interp_index(self.energy_fine, (self.E.tensor - self.bandgap.tensor)/self.gamma.tensor)
        const = 40 * math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy])
        part_3 = 1 - math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy])
        part3_dgamma = const * (self.G[i_theta, i_energy] / (2 * torch.sqrt(self.gamma.tensor)) + (self.bandgap.tensor - self.E.tensor) / self.gamma.tensor ** (3/2) * self.d_energy[i_theta, i_energy])
        part3_dtheta = const * torch.sqrt(self.gamma.tensor) * self.d_theta[i_theta, i_energy]
        part3_dbandgap = const * (-1 / torch.sqrt(self.gamma.tensor)) * self.d_energy[i_theta, i_energy]

        j = [
            # jacobian of error function wrt qfls
            (k * T) ** -1,
            # jacobian of error function wrt gamma
            (part3_dgamma / part_3).unsqueeze(-1),
            # jacobian of error function wrt theta
            (part3_dtheta / part_3).unsqueeze(-1),
            # jacobian of error function wrt bandgap
            (part3_dbandgap / part_3).unsqueeze(-1),
        ]
        return j, self.error()

    def dim(self) -> int:
        return self.I.tensor.shape[1]

    def _copy_impl(self, new_name: Optional[str] = None) -> "QFLSCost":
        return QFLSCost(
            self.weight.copy(), self.E.copy(), self.I.copy(), self.qfls.copy(), self.gamma.copy(), self.theta.copy(), self.bandgap.copy(), name=new_name
        )

batch_size = 10

E_data = torch.linspace(1.65, 1.75, 50).repeat(batch_size, 1)

qfls_data = 1.12*torch.ones(batch_size, 1)
gamma_data = 0.02*torch.ones(batch_size, 1)
theta_data = 1.75*torch.ones(batch_size, 1)
bandgap_data = 1.68*torch.ones(batch_size, 1)

qfls_init = 1.16*torch.ones(batch_size, 1)
gamma_init = 0.015*torch.ones(batch_size, 1)
theta_init = 1.5*torch.ones(batch_size, 1)
bandgap_init = 1.7*torch.ones(batch_size, 1)

qfls = th.Vector(tensor=torch.ones(1), name='qfls')
gamma = th.Vector(tensor=torch.ones(1), name='gamma')
theta = th.Vector(tensor=torch.ones(1), name='theta')
bandgap = th.Vector(tensor=torch.ones(1), name='bandgap')

I_data = generate_gen_planck(E_data, qfls_data, gamma_data, theta_data, bandgap_data)
I_guess = generate_gen_planck(E_data, qfls_init, gamma_init, theta_init, bandgap_init)
I_avg = torch.mean(I_data, dim=0)

fig, ax = plt.subplots()
ax.plot(torch.mean(E_data, dim=0), torch.mean(I_data, dim=0), label='Exact')
ax.plot(torch.mean(E_data, dim=0), torch.mean(I_guess, dim=0), label='Guess')
ax.legend()
plt.show()

# left = next((i for i in np.arange(len(I_avg)) if I_avg[i] > 0.10 * torch.amax(I_avg)), 0)
# right = next((i for i in np.flip(np.arange(len(I_avg))) if I_avg[i] > 0.10 * torch.amax(I_avg)), None)
# E_data = E_data[:, left:right]
# I_data = I_data[:, left:right]

E = th.Variable(E_data, name='E')
I = th.Variable(I_data, name='I')

cost_weight = th.ScaleCostWeight(torch.tensor(1.0))

# construct cost functions and add to objective
objective = th.Objective()
cost_fn = QFLSCost(cost_weight, E, I, qfls, gamma, theta, bandgap)
objective.add(cost_fn)


#%%
error_sq = objective.error_squared_norm()

optimizer = th.LevenbergMarquardt(
    objective,
    max_iterations=25,
    step_size=0.5
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

with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs, optimizer_kwargs={'track_best_solution': True,
                                          'verbose': True})

print('Best solution:', info.best_solution)