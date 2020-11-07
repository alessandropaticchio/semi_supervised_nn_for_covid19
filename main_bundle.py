from training import train_bundle
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import SIRNetwork
from utils import SIR_solution

if __name__ == '__main__':
    # If resume_training is True, it will also load the optimizer and resume training
    resume_training = False

    # Equation parameters
    t_0 = 0
    t_final = 20

    # The intervals in which the equation parameters and the initial conditions should vary
    i_0_set = [0.2, 0.4]
    r_0_set = [0.1, 0.3]
    betas = [0., 0.4]
    gammas = [0.4, 0.7]

    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(i_0_set)
    initial_conditions_set.append(r_0_set)

    # Training parameters
    train_size = 1000
    decay = 1e-3
    hack_trivial = 0
    epochs = 1000
    lr = 8e-4

    # Init model
    sir = SIRNetwork(input=5, layers=4, hidden=50, output=3)

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
                                                              epochs=epochs, model_name=model_name,
                                                              num_batches=10, hack_trivial=hack_trivial,
                                                              train_size=train_size, optimizer=optimizer,
                                                              decay=decay,
                                                              writer=writer, betas=betas, gammas=gammas)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

        import csv
        with open(ROOT_DIR + '/csv/train_losses_{}.csv'.format(model_name), 'w', newline='') as myfile:
            wr = csv.writer(myfile)
            wr.writerow(train_losses)

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    if resume_training:
        additional_epochs = 10000
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        writer = SummaryWriter('runs/' + 'resume_{}'.format(model_name))
        sir, train_losses, run_time, optimizer = train_bundle(sir, initial_conditions_set, t_final=t_final,
                                                              epochs=additional_epochs, model_name=model_name,
                                                              num_batches=10, hack_trivial=hack_trivial,
                                                              train_size=train_size, optimizer=optimizer,
                                                              decay=decay,
                                                              writer=writer, betas=betas, gammas=gammas)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

