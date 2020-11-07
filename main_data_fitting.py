from models import SIRNetwork
from data_fitting import fit
from utils import get_syntethic_data, get_data_dict
from real_data_countries import countries_dict_prelock, countries_dict_postlock, selected_countries_populations, \
    selected_countries_rescaling
from training import train_bundle
from torch.utils.tensorboard import SummaryWriter
from shutil import rmtree
from constants import *
import torch
import os

if __name__ == '__main__':
    # If mode == real, it will fit real data, otherwise synthetic data.
    # Later the data to fit are specified
    mode = 'synthetic'

    t_0 = 0
    t_final = 20

    # The interval in which the equation parameters and the initial conditions should vary
    # i_0_set = [0.4, 0.6]
    # r_0_set = [0.1, 0.3]
    # betas = [0.45, 0.65]
    # gammas = [0.05, 0.15]
    i_0_set = [0.2, 0.4]
    r_0_set = [0.1, 0.3]
    betas = [0., 0.4]
    gammas = [0.4, 0.7]
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)

    # How many times I want to fit the trajectory, getting the best result
    n_trials = 10
    fit_epochs = 300

    # Model parameters
    train_size = 2000
    decay = 1e-4
    hack_trivial = False
    epochs = 3000
    lr = 8e-4

    # Init model
    sir = SIRNetwork(input=5, layers=4, hidden=50)

    model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(i_0_set, r_0_set,
                                                              betas,
                                                              gammas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        writer = SummaryWriter('runs/{}'.format(model_name))
        sir, train_losses, run_time, optimizer = train_bundle(sir, initial_conditions_set, t_final=t_final,
                                                              epochs=epochs,
                                                              num_batches=10, hack_trivial=hack_trivial,
                                                              train_size=train_size, optimizer=optimizer,
                                                              decay=decay,
                                                              writer=writer, betas=betas,
                                                              gammas=gammas)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    writer_dir = 'runs/' + 'fitting_{}'.format(model_name)

    # Check if the writer directory exists, if yes delete it and overwrite
    if os.path.isdir(writer_dir):
        rmtree(writer_dir)
    writer = SummaryWriter(writer_dir)

    if mode == 'real':
        area = 'Italy'
        time_unit = 0.25
        cut_off = 1e-1
        # Real data prelockdown
        data_prelock = get_data_dict(area, data_dict=countries_dict_prelock, time_unit=time_unit,
                                     skip_every=1, cut_off=cut_off, populations=selected_countries_populations,
                                     rescaling=selected_countries_rescaling)
        # Real data postlockdown
        data_postlock = get_data_dict(area, data_dict=countries_dict_postlock, time_unit=time_unit,
                                      skip_every=1, cut_off=0., populations=selected_countries_populations,
                                      rescaling=selected_countries_rescaling)
        susceptible_weight = 1.
        recovered_weight = 1.
        infected_weight = 1.
        force_init = False
    else:
        # Synthetic data
        known_i_0 = 0.25
        known_r_0 = 0.15
        known_beta = 0.2
        known_gamma = 0.5
        data_prelock = get_syntethic_data(sir, t_final=t_final, i_0=known_i_0, r_0=known_r_0, exact_beta=known_beta,
                                          exact_gamma=known_gamma,
                                          size=20)
        susceptible_weight = 1.
        recovered_weight = 1.
        infected_weight = 1.
        force_init = False

    validation_data = {}
    valid_times = []
    valid_infected = []
    valid_recovered = []
    train_val_split = 0.2

    if mode == 'real':
        # Generate validation set by taking the last time units
        max_key = max(data_prelock.keys())
        val_keys = list(data_prelock.keys())

        for k in val_keys:
            if train_val_split == 0.:
                break

            if k >= max_key * (1 - train_val_split):
                valid_times.append(k)
                valid_infected.append(data_prelock[k][1])
                valid_recovered.append(data_prelock[k][2])
                validation_data[k] = [data_prelock[k][0], data_prelock[k][1], data_prelock[k][2]]
                del data_prelock[k]

    else:
        # Generate validation set by sampling equally-spaced points
        max_key = max(data_prelock.keys())
        step = int(1 / train_val_split)
        val_keys = list(data_prelock.keys())[1::step]

        for k in val_keys:
            if train_val_split == 0.:
                break

            valid_times.append(k)
            valid_infected.append(data_prelock[k][1])
            valid_recovered.append(data_prelock[k][2])
            validation_data[k] = [data_prelock[k][0], data_prelock[k][1], data_prelock[k][2]]
            del data_prelock[k]

    min_loss = 1000
    loss_mode = 'mse'
    n_batches = 4

    # Fit n_trials time and take the best fitting
    for i in range(n_trials):
        print('Fit no. {}\n'.format(i + 1))
        i_0, r_0, beta, gamma, val_losses = fit(sir,
                                                init_bundle=initial_conditions_set,
                                                betas=betas,
                                                gammas=gammas,
                                                lr=1e-1,
                                                known_points=data_prelock,
                                                writer=writer,
                                                loss_mode=loss_mode,
                                                epochs=fit_epochs,
                                                verbose=True,
                                                n_batches=n_batches,
                                                susceptible_weight=susceptible_weight,
                                                recovered_weight=recovered_weight,
                                                infected_weight=infected_weight,
                                                force_init=force_init,
                                                validation_data=validation_data)
        s_0 = 1 - (i_0 + r_0)

        if val_losses[-1] < min_loss:
            optimal_s_0, optimal_i_0, optimal_r_0, optimal_beta, optimal_gamma = s_0, i_0, r_0, beta, gamma
            min_loss = val_losses[-1]

    optimal_initial_conditions = [optimal_s_0, optimal_i_0, optimal_r_0]

    print('Estimated initial conditions: S0 = {}, I0 = {}, R0 = {} \n'
          'Estimated Beta = {}, Estimated Gamma = {}'.format(optimal_s_0.item(), optimal_i_0.item(),
                                                             optimal_r_0.item(), optimal_beta.item(),
                                                             optimal_gamma.item()))
