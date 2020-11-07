from models import SIRNetwork
from data_fitting import fit
from utils import get_data_dict
from real_data_countries import countries_dict_prelock, countries_dict_postlock, selected_countries_populations, \
    selected_countries_rescaling
from constants import *
import torch


if __name__ == '__main__':
    t_0 = 0
    t_final = 20

    # The interval in which the equation parameters and the initial conditions should vary

    # Switzerland
    # area = 'Switzerland'
    # i_0_set = [0.01, 0.02]
    # r_0_set = [0.001, 0.006]
    # p_0_set = [0.9, 0.97]
    # betas = [0.7, 0.9]
    # gammas = [0.15, 0.3]

    # Spain
    # area = 'Spain'
    # i_0_set = [0.01, 0.02]
    # r_0_set = [0.004, 0.009]
    # p_0_set = [0.9, 0.97]
    # betas = [0.4, 0.6]
    # gammas = [0.1, 0.2]

    # Italy
    area = 'Italy'
    i_0_set = [0.01, 0.02]
    r_0_set = [0.004, 0.009]
    p_0_set = [0.9, 0.97]
    betas = [0.4, 0.6]
    gammas = [0.1, 0.2]

    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)
    initial_conditions_set.append(p_0_set)

    # How many times I want to fit the trajectory, getting the best result
    n_trials = 10
    fit_epochs = 1000
    n_batches = 10
    fit_lr = 1e-3

    # Model parameters
    train_size = 2000
    decay = 1e-3
    hack_trivial = False
    epochs = 3000
    lr = 8e-4

    # Init model
    sirp = SIRNetwork(input=6, layers=4, hidden=50, output=4)

    model_name = 'i_0={}_r_0={}_p_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set, p_0_set,
                                                                     betas,
                                                                     gammas)

    checkpoint = torch.load(ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(model_name))

    # Load the model
    sirp.load_state_dict(checkpoint['model_state_dict'])

    time_unit = 0.25
    cut_off = 1.5e-3
    multiplication_factor = 10

    # Real data prelockdown
    data_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit,
                                 skip_every=1, cut_off=cut_off, populations=selected_countries_populations,
                                 multiplication_factor=multiplication_factor,
                                 rescaling=selected_countries_rescaling)
    # Real data postlockdown
    data_postlock = get_data_dict(area, data_dict=countries_dict_postlock, time_unit=time_unit,
                                  skip_every=1, cut_off=0., populations=selected_countries_populations,
                                  multiplication_factor=multiplication_factor,
                                  rescaling=selected_countries_rescaling)
    susceptible_weight = 0.
    infected_weight = 1.
    recovered_weight = 1.
    passive_weight = 0.
    force_init = True

    # Generate validation set by taking the last time units
    validation_data = {}
    valid_times = []
    valid_infected = []
    valid_recovered = []
    train_val_split = 0.2
    max_key = max(data_prelock.keys())
    keys = list(data_prelock.keys())

    for k in keys:
        if train_val_split == 0.:
            break

        if k >= max_key * (1 - train_val_split):
            valid_times.append(k)
            valid_infected.append(data_prelock[k][0])
            valid_recovered.append(data_prelock[k][1])
            validation_data[k] = [data_prelock[k][0], data_prelock[k][1]]
            del data_prelock[k]

    min_loss = 10000

    # Fit n_trials time and take the best fitting
    for i in range(n_trials):
        print('Fit no. {}\n'.format(i + 1))
        i_0, r_0, p_0, beta, gamma, val_loss, val_losses = fit(sirp,
                                                               init_bundle=initial_conditions_set,
                                                               betas=betas,
                                                               gammas=gammas,
                                                               lr=fit_lr,
                                                               known_points=data_prelock,
                                                               mode='real',
                                                               epochs=fit_epochs,
                                                               verbose=True,
                                                               n_batches=n_batches,
                                                               susceptible_weight=susceptible_weight,
                                                               infected_weight=infected_weight,
                                                               recovered_weight=recovered_weight,
                                                               passive_weight=passive_weight,
                                                               force_init=force_init,
                                                               validation_data=validation_data)
        s_0 = 1 - (i_0 + r_0 + p_0)

        if val_loss < min_loss:
            optimal_s_0, optimal_i_0, optimal_r_0, optimal_p_0, optimal_beta, optimal_gamma = s_0, i_0, r_0, p_0, beta, gamma
            min_loss = val_loss

    optimal_initial_conditions = [optimal_s_0, optimal_i_0, optimal_r_0, optimal_p_0]



    print('Estimated initial conditions: S0 = {}, I0 = {}, R0 = {}, P0 = {} \n'
          'Estimated Beta = {}, Estimated Gamma = {}'.format(optimal_s_0.item(), optimal_i_0.item(),
                                                             optimal_r_0.item(), optimal_p_0.item(), optimal_beta.item(),
                                                             optimal_gamma.item()))
