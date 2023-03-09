from typing import List, Optional, Tuple
import pkg_resources
import theseus as th
import torch
import math
import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.optimize import curve_fit
from scipy.interpolate import RectBivariateSpline

torch.manual_seed(0)
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = 'cpu'

h = 4.1357e-15
c = 2.9979e8
k = 8.6173e-5
q = 1.602e-19
T = 298.15
alpha = 5
THETA = np.linspace(0.5, 2, round(1.5 / 0.05) + 1)  # Values taken from appropriate endpoints of lookup table
ENERGY = np.linspace(-60, 100, round((100 + 60) / 0.1) + 1)  # Values taken from appropriate endpoints of lookup table

LOOKUP_TABLE = pkg_resources.resource_stream(__name__, 'data/lookup_table.csv')


def generate_integral_table(filename):
    integrals = pd.read_csv(LOOKUP_TABLE, names=['G'], usecols=[2])
    integrals = np.reshape(integrals['G'].to_numpy(), (THETA.size, ENERGY.size))
    SPLINE_INTERP = RectBivariateSpline(THETA, ENERGY, integrals, kx=3, ky=3, s=0)


def import_hyperspectral(filename):
    with h5py.File(filename, 'r') as f:
        wavelength = np.array(f['Cube/Wavelength'], dtype=np.float32)
        cube = np.asarray(f['Cube/Images'], dtype=np.float32)
    cube = np.clip(cube, 0, None)
    energy = 1E9 * h * c / np.flip(wavelength, axis=0)
    cube = np.flip(cube, axis=0)
    cube = 10000 * math.pi * cube
    return energy, cube


def crop_data(energy, pl):
    """ Crops energy axis to only include data points above 10% of peak maximum

    :param energy: Photon energy (eV)
    :param pl: Photoluminescence intensity
    :return: Cropped energy and photoluminescence
    """
    pl_avg = np.average(pl, axis=(1, 2))
    low = next((i for i in np.arange(len(pl_avg)) if pl_avg[i] > 0.10 * np.amax(pl_avg)), 0)
    high = next((i for i in np.flip(np.arange(len(pl_avg))) if pl_avg[i] > 0.10 * np.amax(pl_avg)), None)
    return energy[low:high], pl[low:high, :, :]


def eval_interp(delta_e, theta):
    """Flips scaled energy axis such that it is monotonically increasing for spline interpolation

    :param delta_e: Scaled energy axis
    :param theta: Functional form of sub-bandgap absorption
    :return: Interpolated integrals
    """
    delta_e = delta_e.flatten()
    if delta_e[-1] >= delta_e[0]:
        g = SPLINE_INTERP(theta, delta_e).flatten()
    else:
        g = np.flip(SPLINE_INTERP(theta, np.flip(delta_e)).flatten())
    return g


def gen_lookup_table(accuracy_factor=10):
    """Generates tensor of interpolated integral values

    :return: Torch tensors for theta, gamma, the partial derivative of G w.r.t theta, the partial derivative of G w.r.t
        gamma, and the values for G.
    """
    theta_fine = np.linspace(np.min(THETA), np.max(THETA), accuracy_factor * THETA.size)
    energy_fine = np.linspace(np.min(ENERGY), np.max(ENERGY), accuracy_factor * ENERGY.size)
    d_theta = torch.tensor(SPLINE_INTERP.partial_derivative(1, 0)(theta_fine, energy_fine), dtype=torch.float32)
    d_energy = torch.tensor(SPLINE_INTERP.partial_derivative(0, 1)(theta_fine, energy_fine), dtype=torch.float32)
    G_tensor = torch.tensor(SPLINE_INTERP(theta_fine, energy_fine), dtype=torch.float32)
    return torch.tensor(theta_fine, dtype=torch.float32), torch.tensor(energy_fine,
                                                                       dtype=torch.float32), d_theta, d_energy, G_tensor


def gen_planck(energy, qfls, gamma, theta, bandgap):
    """Evaluates generalized Planck equation according to input parameters

    :param energy: Photon energy (eV)
    :param qfls: Quasi-Fermi level splitting (eV)
    :param gamma: Sub-bandgap absorption broadening energy
    :param theta: Functional form of sub-bandgap absorption
    :param bandgap: Bandgap (eV)
    :return: Photoluminescence spectrum given by generalized Planck equation
    """
    part1 = 2 * math.pi * energy ** 2 / (h ** 3 * c ** 2)
    part2 = 1 / (math.e ** ((energy - qfls) / (k * T)) - 1)
    part3 = (1 - math.e ** (-alpha * np.sqrt(gamma) * eval_interp((energy - bandgap) / gamma, theta)))
    return (part1 * part2 * part3).squeeze()


def interp_index(arr, val):
    """Returns the indices of elements of an array closest to the input values

    :param arr: Input array
    :param val: Input values
    :return: Indices of closest elements
    """
    arr_min = val.unsqueeze(-1) - arr.unsqueeze(0)
    values, indices = torch.min(torch.abs(arr_min), -1)
    return indices


def fit_qfls(energy, pl, guesses=(1.1, 0.01, 1.5, 1.6), batch_size=56, max_iterations=20, step_size=0.5, accuracy=4, verbose=True):
    """Fits input hyperspectral data according to the generalized Planck equation

    :param energy: Input photon energy (eV)
    :param pl: Experimental absolute-intensity photoluminescence (m^-2 s^-1 eV^-1)
    :param guesses: Guessed quasi-Fermi level splitting, gamma, theta, and bandgap
    :return: Dictionary containing all fitted parameters for image
    """
    energy_tensor = torch.tensor(energy[None, :], dtype=torch.float32, device=device)
    pl_reshape = pl.reshape((pl.shape[0], -1)).T
    pl_tensor = torch.tensor(pl_reshape, dtype=torch.float32, device=device)
    guess_tensors = [guesses[i] * torch.ones((batch_size, 1), dtype=torch.float32) for i in range(len(guesses))]
    result = {'qfls': np.zeros(pl_reshape.shape[0], dtype=np.float32),
              'gamma': np.zeros(pl_reshape.shape[0], dtype=np.float32),
              'theta': np.zeros(pl_reshape.shape[0], dtype=np.float32),
              'bandgap': np.zeros(pl_reshape.shape[0], dtype=np.float32)}
    n_loops = round(pl_reshape.shape[0] / batch_size)
    inputs = {'energy': energy_tensor.to(device),
              'pl': pl_tensor[0:batch_size, :].to(device),
              'qfls': guess_tensors[0].to(device),
              'gamma': guess_tensors[1].to(device),
              'theta': guess_tensors[2].to(device),
              'bandgap': guess_tensors[3].to(device)}
    optimizer, objective = init_optimizer(inputs, batch_size, energy.size, max_iterations, step_size, accuracy)
    error_sq = objective.error_squared_norm()
    with torch.no_grad():
        for i in tqdm(range(n_loops)):
            inputs['pl'] = pl_tensor[i * batch_size:(i + 1) * batch_size, :]
            objective.update(inputs)
            updated_inputs, info = optimizer.forward(
                inputs, optimizer_kwargs={'track_best_solution': True,
                                          'track_err': True,
                                          'verbose': verbose
                                          })
            for key in result.keys():
                result[key][i * batch_size:(i + 1) * batch_size] = info.best_solution[key].numpy().flatten()
    for key in result.keys():
        result[key] = result[key].reshape((pl.shape[1], pl.shape[2]))
    return result, info


def init_optimizer(inputs, batch_size, data_size, max_iterations, step_size, accuracy):
    energy = th.Variable(torch.ones((batch_size, data_size), dtype=torch.float32), name='energy')
    energy.to(device)
    pl = th.Variable(torch.ones((batch_size, data_size), dtype=torch.float32), name='pl')
    pl.to(device)
    qfls = th.Vector(1, name='qfls', dtype=torch.float32)
    qfls.to(device)
    gamma = th.Vector(1, name='gamma', dtype=torch.float32)
    gamma.to(device)
    theta = th.Vector(1, name='theta', dtype=torch.float32)
    theta.to(device)
    bandgap = th.Vector(1, name='bandgap', dtype=torch.float32)
    bandgap.to(device)

    cost_weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float32, device=device))
    objective = th.Objective(dtype=torch.float32)
    objective.to(device=device)
    cost_fn = QFLSCost(cost_weight, energy, pl, qfls, gamma, theta, bandgap, accuracy=accuracy)
    objective.add(cost_fn)
    objective.update(inputs)
    error_sq = objective.error_squared_norm()
    optimizer = th.TheseusLayer(th.LevenbergMarquardt(objective, max_iterations=max_iterations, step_size=step_size))
    optimizer.to(device=device)
    return optimizer, objective


class QFLSCost(th.CostFunction):
    """Custom cost function for fitting to generalized Planck equation

    """

    def __init__(
            self,
            cost_weight: th.CostWeight,
            energy: th.Variable,
            pl: th.Variable,
            qfls: th.Vector,
            gamma: th.Vector,
            theta: th.Vector,
            bandgap: th.Vector,
            name: Optional[str] = None,
            accuracy: Optional[int] = 4
    ):
        """Initializes QFLSCost with auxiliary and optimization variables

        :param cost_weight: Used to weight errors and jacobians
        :param energy: Photon energy (eV) (auxiliary variable)
        :param pl: Photoluminescence intensity (m^-2 s^-1 eV^-1) (auxiliary variable)
        :param qfls: Quasi-Fermi level splitting (eV)
        :param gamma: Sub-bandgap absorption broadening energy
        :param theta: Functional form of sub-bandgap absorption
        :param bandgap: Bandgap (eV)
        :param name: Optional name assigned to cost function
        """
        super().__init__(cost_weight, name=name)
        self.energy = energy
        self.pl = pl
        self.qfls = qfls
        self.gamma = gamma
        self.theta = theta
        self.bandgap = bandgap

        lookup = gen_lookup_table(accuracy_factor=accuracy)
        self.theta_fine, self.energy_fine, self.d_theta, self.d_energy, self.G = [arr.to(device) for arr in lookup]

        self.register_optim_vars(['qfls', 'gamma', 'theta', 'bandgap'])
        self.register_aux_vars(['energy', 'pl'])

        self.part1 = 2 * math.pi * self.energy.tensor ** 2 / (h ** 3 * c ** 2) * math.e ** (-self.energy.tensor / (k * T))

    def gen_planck(self):
        """Evaluates generalized Planck equation according to attribute parameters

        :return: Photoluminescence spectrum
        """
        i_theta = interp_index(self.theta_fine, self.theta.tensor)
        i_energy = interp_index(self.energy_fine, (self.energy.tensor - self.bandgap.tensor) / self.gamma.tensor)
        return torch.squeeze(
            # Part 1
            torch.log(2 * math.pi * self.energy.tensor ** 2 / (h ** 3 * c ** 2)) - self.energy.tensor / (k * T) +
            # Part 2
            self.qfls.tensor / (k * T) +
            # Part 3
            torch.log(1 - math.e ** (-alpha * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]))
        )

    def error(self) -> torch.Tensor:
        return self.gen_planck() - torch.log(self.pl.tensor)

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Calculates all first-order partial derivatives of multivariate error function

        :return: Jacobian matrix of error function, error function
        """
        i_theta = interp_index(self.theta_fine, self.theta.tensor)
        i_energy = interp_index(self.energy_fine, (self.energy.tensor - self.bandgap.tensor) / self.gamma.tensor)
        const = alpha * math.e ** (-alpha * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy])
        part_3 = 1 - math.e ** (-alpha * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy])
        part3_dgamma = const * (self.G[i_theta, i_energy] / (2 * torch.sqrt(self.gamma.tensor)) + (
                    self.bandgap.tensor - self.energy.tensor) / self.gamma.tensor ** (3 / 2) * self.d_energy[
                                    i_theta, i_energy])
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
        return self.pl.tensor.shape[1]

    def _copy_impl(self, new_name: Optional[str] = None) -> "QFLSCost":
        return QFLSCost(
            self.weight.copy(), self.energy.copy(), self.pl.copy(), self.qfls.copy(), self.gamma.copy(), self.theta.copy(),
            self.bandgap.copy(), name=new_name
        )
