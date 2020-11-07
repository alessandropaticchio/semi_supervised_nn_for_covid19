import numpy as np
from datetime import timedelta
from torch.utils.data import DataLoader
from scipy.integrate import odeint
from constants import *

def perturb_points(grid, t_0, t_final, sig=0.5):
    # Stochastic perturbation of the evaluation points
    # Force t[0]=t0 and force points to be in the t-interval
    delta_t = grid[1] - grid[0]
    noise = delta_t * torch.randn_like(grid) * sig
    t = grid + noise
    t.data[2] = torch.ones(1, 1) * (-1)
    t.data[t < t_0] = t_0 - t.data[t < t_0]
    t.data[t > t_final] = 2 * t_final - t.data[t > t_final]
    t.data[0] = torch.ones(1, 1) * t_0
    t.requires_grad = False
    return t


def generate_dataloader(grid, t_0, t_final, batch_size, perturb=True, shuffle=True):
    # Generate a dataloader with perturbed points starting from a grid_explorer
    if perturb:
        grid = perturb_points(grid, t_0, t_final, sig=0.3 * t_final)
    grid.requires_grad = True

    t_dl = DataLoader(dataset=grid, batch_size=batch_size, shuffle=shuffle)
    return t_dl


# Use below in the Scipy Solver
def f_sirp(u, t, beta, gamma):
    s, i, r, p = u  # unpack current values of u

    N = s + i + r

    derivs = [-(beta * i * s) / N, (beta * i * s) / N - gamma * i, gamma * i, 0]  # list of dy/dt=f functions

    return derivs


# Scipy Solver
def SIRP_solution(t, s_0, i_0, r_0, p_0, beta, gamma):
    u_0 = [s_0, i_0, r_0, p_0]

    # Call the ODE solver
    sol_sir = odeint(f_sirp, u_0, t, args=(beta, gamma))
    s = sol_sir[:, 0]
    i = sol_sir[:, 1]
    r = sol_sir[:, 2]
    p = sol_sir[:, 3]

    return s, i, r, p


### END SEIR MODEL


# Function to sample synthetic data from a generic solution of a model
def get_syntethic_data(model, t_final, i_0, r_0, p_0, exact_beta, exact_gamma, size):
    model.eval()

    s_0 = 1 - (i_0 + r_0 + p_0)

    # Generate tensors and get known points from the ground truth
    exact_initial_conditions = [torch.Tensor([s_0]).reshape(-1, 1),
                                torch.Tensor([i_0]).reshape(-1, 1),
                                torch.Tensor([r_0]).reshape(-1, 1), torch.Tensor([p_0]).reshape(-1, 1)]

    exact_beta = torch.Tensor([exact_beta]).reshape(-1, 1)
    exact_gamma = torch.Tensor([exact_gamma]).reshape(-1, 1)

    synthetic_data = {}

    rnd_t = np.linspace(0, t_final - 1, size)

    for t in rnd_t:
        t = torch.Tensor([t])
        t = t.reshape(-1, 1)
        s_p, i_p, r_p, p_p = model.parametric_solution(t, exact_initial_conditions,
                                                  beta=exact_beta,
                                                  gamma=exact_gamma,)
        synthetic_data[t.item()] = [s_p.item(), i_p.item(), r_p.item(), p_p.item()]

    return synthetic_data

def generate_grid(t_0, t_final, size):
    grid = torch.linspace(t_0, t_final, size).reshape(-1, 1)
    return grid


def get_data_dict(area, data_dict, time_unit, populations, rescaling, scaled=True, skip_every=None, cut_off=1e-3,
                  multiplication_factor=1, return_new_cases=False, reducing_population=False, return_cut_off_date=False):
    """
    :param area: name of the area where I want to extrapolate the data
    :param data_dict: dictionary that contains the data for a given area
    :param time_unit: time unit I want to use in my system. 1 unit = (1 / time_unit) days
    :param skip_every: if not None, data will be sampled once every skip_every days
    :param multiplication_factor: total cases and recovered are multiplied by this factor
    :param cut_off: minimum amount of infected to start sampling
    :param return_new_cases: if True, it returns the trend of new cases
    :return: if return_new_cases is False, it returns a dictionary mapping t to [s(t), i(t), r(t)]
    """

    if area in rescaling.keys() and reducing_population:
        population = populations[area] * rescaling[area]
    else:
        population = populations[area]

    # A dictionary that will collect the data as
    # t : [s(t), i(t), r(t)]
    traj = {}

    # Select only the days whose infected go over the minimum cut off
    for_cut_off_checking = np.array(data_dict[area][0])
    if scaled:
         for_cut_off_checking = for_cut_off_checking / populations[area]

    if cut_off == 0:
        d = 0
    else:
        for d, i in enumerate(for_cut_off_checking):
            if i > cut_off:
                break

    area_infected = np.array(data_dict[area][0][d:])
    area_removed = np.array(data_dict[area][1][d:])
    area_new_cases = data_dict[area][2][d:]

    # Get in what day the population reached the cut off
    cut_off_day = data_start + timedelta(days=d)

    # Going from active cases to cumulated cases and rescaling by a given factor
    area_infected = ((area_infected + area_removed) * multiplication_factor)

    # Rescaling by a given factor
    area_removed = area_removed * multiplication_factor

    # Going back to active cases
    area_infected = area_infected - area_removed

    # Rescale infected and removed between 0 and 1
    if scaled:
        area_infected = np.array(area_infected) / population
        area_removed = np.array(area_removed) / population

    times = []

    for i in range(len(area_infected)):
        times.append(i * time_unit)
        traj[i * time_unit] = [area_infected[i], area_removed[i]]

    # If I don't want to select contiguous day, I will get just a subset
    if skip_every:

        keys = list(traj.keys())
        for i, j in enumerate(keys):
            if i % skip_every != 0:
                del traj[j]

    if return_new_cases:
        return times, area_new_cases
    else:
        if return_cut_off_date:
            return traj, cut_off_day
        else:
            return traj