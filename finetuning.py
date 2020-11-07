from constants import ROOT_DIR
from models import SIRNetwork
from training import train_bundle
from torch.utils.tensorboard import SummaryWriter
import torch

if __name__ == '__main__':
    # File to apply finetuning on a pretrained model

    source_i_0_set = [0.1, 0.2]
    source_r_0_set = [0.1, 0.2]
    source_betas = [0.6, 0.8]
    source_gammas = [0.1, 0.2]

    initial_conditions_set = []
    t_0 = 0
    t_final = 20
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(source_i_0_set)
    initial_conditions_set.append(source_r_0_set)
    # Init model
    sir = SIRNetwork(input=5, layers=4, hidden=50)
    lr = 8e-4

    source_model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(source_i_0_set, source_r_0_set,
                                                                     source_betas,
                                                                     source_gammas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(ROOT_DIR + '/models/SIR_bundle_total/{}'.format(source_model_name))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        source_epochs = 20000
        source_hack_trivial = 0
        source_train_size = 2000
        source_decay = 1e-2
        writer = SummaryWriter('runs/{}_scratch'.format(source_model_name))
        sir, train_losses, run_time, optimizer = train_bundle(sir, initial_conditions_set, t_final=t_final,
                                                              epochs=source_epochs, model_name=source_model_name,
                                                              num_batches=10, hack_trivial=source_hack_trivial,
                                                              train_size=source_train_size, optimizer=optimizer,
                                                              decay=source_decay,
                                                              writer=writer, betas=source_betas,
                                                              gammas=source_gammas)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/{}'.format(source_model_name))

        # Load the checkpoint
        checkpoint = torch.load(ROOT_DIR + '/models/SIR_bundle_total/{}'.format(source_model_name))

    # Target model
    target_i_0_set = [0.05, 0.15]
    target_r_0_set = [0.01, 0.03]
    target_betas = [0.45, 0.60]
    target_gammas = [0.05, 0.15]

    target_model_name = 'i_0={}_r_0={}_betas={}_gammas={}.pt'.format(target_i_0_set, target_r_0_set,
                                                                     target_betas,
                                                                     target_gammas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(target_model_name))

    except FileNotFoundError:
        print('Finetuning...')
        # Load old model
        sir.load_state_dict(checkpoint['model_state_dict'])
        # Train
        initial_conditions_set = []
        t_0 = 0
        t_final = 20
        initial_conditions_set.append(t_0)
        initial_conditions_set.append(target_i_0_set)
        initial_conditions_set.append(target_r_0_set)
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        target_epochs = 10000
        target_hack_trivial = 0
        target_train_size = 2000
        target_decay = 1e-3
        writer = SummaryWriter('runs/{}_finetuned'.format(target_model_name))
        sir, train_losses, run_time, optimizer = train_bundle(sir, initial_conditions_set, t_final=t_final,
                                                              epochs=target_epochs, model_name=target_model_name,
                                                              num_batches=10, hack_trivial=target_hack_trivial,
                                                              train_size=target_train_size, optimizer=optimizer,
                                                              decay=target_decay,
                                                              writer=writer, betas=target_betas,
                                                              gammas=target_gammas)

        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/{}'.format(target_model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/{}'.format(target_model_name))