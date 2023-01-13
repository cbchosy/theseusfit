import torch
import theseus as th
import numpy as np

arr_size = int(16)

torch.manual_seed(0)
data_x = torch.tensor(np.stack([np.linspace(0, 1, 100)]*arr_size), dtype=torch.float32)
data_coeffs = torch.tensor(np.random.random(data_x.shape[0]), dtype=torch.float32)
data_y = data_coeffs.unsqueeze(-1)*torch.exp(data_x)

interp_x_data = torch.linspace(0, 1, 100).unsqueeze(0)
interp_y_data = torch.pow(interp_x_data, 1) + torch.pow(interp_x_data, 2) + torch.pow(interp_x_data, 3)

x = th.Variable(data_x, name='x')
y = th.Variable(data_y, name='y')
interp_x = th.Variable(interp_x_data, name='interp_x')
interp_y = th.Variable(interp_y_data, name='interp_y')

a = th.Vector(1, name='a')

def interp_index(arr, val):
    arr_min = val - arr
    values, indices = torch.min(torch.abs(arr_min), -1)
    return indices

def error_fn(optim_vars, aux_vars):
    a, = optim_vars
    x, y, interp_x, interp_y = aux_vars
    coeff = torch.pow(a.tensor, 1) + torch.pow(a.tensor, 2) + torch.pow(a.tensor, 3)
    est = coeff*torch.exp(x.tensor)
    err = y.tensor - est
    return err

optim_vars = a,
aux_vars = x, y, interp_x, interp_y
cost_function = th.AutoDiffCostFunction(
    optim_vars, error_fn, 100,
    aux_vars=aux_vars, name='cost_fn'
)
objective = th.Objective()
objective.add(cost_function)
optimizer = th.GaussNewton(
    objective,
    max_iterations=15,
    step_size=0.5
)
theseus_optim = th.TheseusLayer(optimizer)

theseus_inputs = {
    'x': data_x,
    'y': data_y,
    'interp_x': interp_x_data,
    'interp_y': interp_y_data,
    'a': torch.ones((arr_size, 1))
}
with torch.no_grad():
    updated_inputs, info = theseus_optim.forward(
        theseus_inputs,
        optimizer_kwargs={'track_best_solution': True, 'verbose': True}
    )
print('Best solution:', info.best_solution)
coeffs_est = np.array(info.best_solution['a'])
coeffs_exact = np.array(data_coeffs)