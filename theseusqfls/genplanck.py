from typing import List, Optional, Tuple
import pkg_resources
import theseus as th
import torch
import math
import h5py
import numpy as np
import pandas as pd
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
alpha = 10

LOOKUP_TABLE = "C:/Users/cbc37/Desktop/theseusqfls/theseusqfls/data/lookup_table.csv"
# STREAM = pkg_resources.resource_stream(__name__, LOOKUP_TABLE)
THETA = np.linspace(0.5, 2, round(1.5 / 0.05) + 1)  # Values taken from appropriate endpoints of lookup table
ENERGY = np.linspace(-60, 100, round((100 + 60) / 0.1) + 1)  # Values taken from appropriate endpoints of lookup table
integrals = pd.read_csv(LOOKUP_TABLE, names=['G'], usecols=[2])
integrals = np.reshape(integrals['G'].to_numpy(), (THETA.size, ENERGY.size))
SPLINE_INTERP = RectBivariateSpline(THETA, ENERGY, integrals, kx=3, ky=3, s=0)

# def interpolate_integral():
#     """Generates table of integrals used in evaluating the generalized Planck equation.
#
#     Pre-calculated values are read from lookup_table.csv and then interpolated to reach the appropriate accuracy. The
#     table is expressed as a function of theta (which determines the functional form of the subgap tail) and gamma
#     (the broadening energy of the subgap tail).
#
#     :return: Spline interpolation of integral values
#     """
#     integrals = pd.read_csv(LOOKUP_TABLE, names=['G'], usecols=[2])
#     integrals = np.reshape(integrals['G'].to_numpy(), (THETA.size, ENERGY.size))
#     return RectBivariateSpline(THETA, ENERGY, integrals, kx=3, ky=3, s=0)


def import_hyperspectral(filename):
    with h5py.File(filename, 'r') as f:
        wavelength = np.array(f['Cube/Wavelength'], dtype=np.float32)
        cube = np.asarray(f['Cube/Images'], dtype=np.float32)
    cube = np.clip(cube, 0, None)
    energy = 1E9 * h * c / np.flip(wavelength, axis=0)
    cube = np.flip(cube, axis=0)  # TODO add Jacobian transformation here
    cube = 10000 * math.pi * cube
    return energy, cube


def eval_interp(delta_e, theta):
    """Flips scaled energy axis such that it is monotonically increasing for spline interpolation

    :param delta_e: Scaled energy axis
    :param theta: Functional form of sub-bandgap absorption
    :return: Interpolated integrals
    """
    delta_e = delta_e.flatten()
    if delta_e[-1] >= delta_e[0]:
        g = SPLINE_INTERP(theta, delta_e)
    else:
        g = np.flip(SPLINE_INTERP(theta, np.flip(delta_e)))
    return g


def gen_lookup_table():
    """Generates tensor of interpolated integral values

    :return: Torch tensors for theta, gamma, the partial derivative of G w.r.t theta, the partial derivative of G w.r.t
        gamma, and the values for G.
    """
    theta_fine = np.linspace(np.min(THETA), np.max(THETA), 50*THETA.size)
    energy_fine = np.linspace(np.min(ENERGY), np.max(ENERGY), 50*ENERGY.size)
    d_theta = torch.tensor(SPLINE_INTERP.partial_derivative(1, 0)(theta_fine, energy_fine), dtype=torch.float32)
    d_energy = torch.tensor(SPLINE_INTERP.partial_derivative(0, 1)(theta_fine, energy_fine), dtype=torch.float32)
    G_tensor = torch.tensor(SPLINE_INTERP(theta_fine, energy_fine), dtype=torch.float32)
    return torch.tensor(theta_fine), torch.tensor(energy_fine), d_theta, d_energy, G_tensor


# def generate_gen_planck(E, qfls, gamma, theta, bandgap, noise_factor=0.0):
#     """Generates simulated photoluminescence spectra according to the generalized Planck equation.
#
#     :param E: Photon energy (eV)
#     :param qfls: Quasi-Fermi level splitting (eV)
#     :param gamma: Sub-bandgap absorption broadening energy
#     :param theta: Functional form of sub-bandgap absorption
#     :param bandgap: Bandgap (eV)
#     :param noise_factor: Introduces random noise on top of simulated PL signal
#     :return: Simulated photoluminescence spectrum as a function of the photon energy, E
#     """
#     theta_fine, energy_fine, d_theta, d_energy, G = gen_lookup_table()
#     i_theta = interp_index(theta_fine, theta)
#     i_energy = interp_index(energy_fine, (E - bandgap) / gamma)
#     part1 = 2 * math.pi * E ** 2 / (h ** 3 * c ** 2) * math.e ** (-E / (k * T))
#     part2 = math.e ** (qfls / (k * T))
#     part3 = (1 - math.e ** (-40 * torch.sqrt(gamma) * G[i_theta, i_energy]))
#     return part1 * part2 * part3 + torch.randn(part3.shape) * noise_factor


def gen_planck(energy, qfls, gamma, theta, bandgap):
    """Evaluates generalized Planck equation according to input parameters

    :param energy: Photon energy (eV)
    :param qfls: Quasi-Fermi level splitting (eV)
    :param gamma: Sub-bandgap absorption broadening energy
    :param theta: Functional form of sub-bandgap absorption
    :param bandgap: Bandgap (eV)
    :return: Photoluminescence spectrum given by generalized Planck equation
    """
    part1 = 2*math.pi*energy**2/(h**3*c**2)
    part2 = 1/(math.e**((energy - qfls)/(k*T)) - 1)
    part3 = (1 - math.e**(-40*np.sqrt(gamma)*eval_interp((energy - bandgap)/gamma, theta)))
    part4 = (1 - 2/(math.e**((energy - qfls)/(2*k*T)) + 1))
    f = np.array(part1*part2*part3*part4).squeeze()
    return f


def interp_index(arr, val):
    """Returns the indices of elements of an array closest to the input values

    :param arr: Input array
    :param val: Input values
    :return: Indices of closest elements
    """
    arr_min = val.unsqueeze(-1) - arr.unsqueeze(0)
    values, indices = torch.min(torch.abs(arr_min), -1)
    return indices


def fit_qfls(energy, pl, guesses=None):
    """Fits input hyperspectral data according to the generalized Planck equation

    :param energy: Input photon energy (eV)
    :param pl: Experimental absolute-intensity photoluminescence (m^-2 s^-1 eV^-1)
    :param guesses: Guessed quasi-Fermi level splitting, gamma, theta, and bandgap
    :return: Dictionary containing all fitted parameters for image
    """
    if guesses is None:
        guesses = [1.1, 0.05, 1.5, 1.6]
        pl_avg = np.expand_dims(np.mean(pl, axis=(1, 2)), axis=(1, 2))
        try:
            guesses, __ = curve_fit(gen_planck, energy, pl_avg, p0=guesses)
        except:
            pass
    energy_tensor = torch.tensor(energy)
    pl_tensor = torch.tensor(np.flatten(pl))
    guess_tensors = [guesses[i]*torch.ones(pl_tensor.shape, 1, dtype=torch.float64) for i in range(len(guesses))]

    qfls = th.Vector(1, name='qfls', dtype=torch.float32)
    gamma = th.Vector(1, name='gamma', dtype=torch.float32)
    theta = th.Vector(1, name='theta', dtype=torch.float32)
    bandgap = th.Vector(1, name='bandgap', dtype=torch.float32)

    cost_weight = th.ScaleCostWeight(torch.tensor(1.0, dtype=torch.float64))
    objective = th.Objective(dtype=torch.float64)
    cost_fn = QFLSCost(cost_weight, th.Variable(energy_tensor, name='energy'), th.Variable(pl_tensor, name='pl'), qfls,
                       gamma, theta, bandgap)
    objective.add(cost_fn)
    error_sq = objective.error_squared_norm()
    optimizer = th.LevenbergMarquardt(objective, max_iterations=2, step_size=0.1)
    theseus_optim = th.TheseusLayer(optimizer)
    theseus_optim.to(device=device)  # TODO check correct data type here

    # Send all inputs to GPU if available
    theseus_inputs = {'energy': energy_tensor.to(device),
                      'pl': pl_tensor.to(device),
                      'qfls': guess_tensors[0].to(device),
                      'gamma': guess_tensors[1].to(device),
                      'theta': guess_tensors[2].to(device),
                      'bandgap': guess_tensors[3].to(device)}
    objective.update(theseus_inputs)

    with torch.no_grad():
        updated_inputs, info = theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={'track_best_solution': True, 'verbose': True})
    return info.best_solution


class QFLSCost(th.CostFunction):
    """Custom cost function for fitting to generalized Planck equation

    """

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
        """Initializes QFLSCost with auxiliary and optimization variables

        :param cost_weight: Used to weight errors and jacobians
        :param E: Photon energy (eV) (auxiliary variable)
        :param I: Photoluminescence intensity (m^-2 s^-1 eV^-1) (auxiliary variable)
        :param qfls: Quasi-Fermi level splitting (eV)
        :param gamma: Sub-bandgap absorption broadening energy
        :param theta: Functional form of sub-bandgap absorption
        :param bandgap: Bandgap (eV)
        :param name: Optional name assigned to cost function
        """
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
        """Evaluates generalized Planck equation according to attribute parameters

        :return: Photoluminescence spectrum
        """
        i_theta = interp_index(self.theta_fine, self.theta.tensor)
        i_energy = interp_index(self.energy_fine, (self.E.tensor - self.bandgap.tensor) / self.gamma.tensor)
        part1 = 2 * math.pi * self.E.tensor ** 2 / (h ** 3 * c ** 2) * math.e ** (-self.E.tensor / (k * T))
        part2 = math.e ** (self.qfls.tensor / (k * T))
        part3 = (1 - math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]))
        return torch.squeeze(part1 * part2 * part3)

    def error(self) -> torch.Tensor:
        return self.gen_planck() - self.I.tensor

    def jacobians(self) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Calculates all first-order partial derivatives of multivariate error function

        :return: Jacobian matrix of error function, error function
        """
        i_theta = interp_index(self.theta_fine, self.theta.tensor)
        i_energy = interp_index(self.energy_fine, (self.E.tensor - self.bandgap.tensor) / self.gamma.tensor)
        part1 = 2 * math.pi * self.E.tensor ** 2 / (h ** 3 * c ** 2) * math.e ** (-self.E.tensor / (k * T))
        part2 = math.e ** (self.qfls.tensor / (k * T))
        part3 = (1 - math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]))

        part3_dgamma = 40 * math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]) * (
                    self.G[i_theta, i_energy] / (2 * torch.sqrt(self.gamma.tensor)) + (
                        self.bandgap.tensor - self.E.tensor) / self.gamma.tensor ** (3 / 2) * self.d_energy[
                        i_theta, i_energy])
        part3_dtheta = 40 * math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]) * torch.sqrt(
            self.gamma.tensor) * self.d_theta[i_theta, i_energy]
        part3_dbandgap = 40 * math.e ** (-40 * torch.sqrt(self.gamma.tensor) * self.G[i_theta, i_energy]) * (
                    -1 / torch.sqrt(self.gamma.tensor)) * self.d_energy[i_theta, i_energy]

        j = [
            # Partial derivative of error function wrt qfls
            ((k * T) ** -1 * part1 * part2 * part3).unsqueeze(-1),
            # Partial derivative of error function wrt gamma
            (part1 * part2 * part3_dgamma).unsqueeze(-1),
            # Partial derivative of error function wrt theta
            (part1 * part2 * part3_dtheta).unsqueeze(-1),
            # Partial derivative of error function wrt bandgap
            (part1 * part2 * part3_dbandgap).unsqueeze(-1),
        ]
        return j, self.error()

    def dim(self) -> int:
        return self.I.tensor.shape[1]

    def _copy_impl(self, new_name: Optional[str] = None) -> "QFLSCost":
        return QFLSCost(
            self.weight.copy(), self.E.copy(), self.I.copy(), self.qfls.copy(), self.gamma.copy(), self.theta.copy(),
            self.bandgap.copy(), name=new_name
        )
if __name__ == '__main__':
    energy, pl = import_hyperspectral("C:/Users/cbc37/Desktop/theseusqfls/theseusqfls/data/example_photometric.h5")
    result = fit_qfls(energy, pl)
