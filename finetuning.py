from constants import ROOT_DIR
from models import SIRNetwork
from training import train_bundle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch

if __name__ == '__main__':
    # File to apply finetuning on a pretrained model

    # Equation parameters
    t_0 = 0
    t_final = 20

    # The intervals in which the equation parameters and the initial conditions should vary
    source_i_0_set = [0.01, 0.02]
    source_r_0_set = [0.004, 0.009]
    source_p_0_set = [0.9, 0.97]
    source_betas = [0.5, 0.7]
    source_gammas = [0.1, 0.2]

    # Model parameters
    initial_conditions_set = []
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(source_i_0_set)
    initial_conditions_set.append(source_r_0_set)
    initial_conditions_set.append(source_p_0_set)

    train_size = 1000
    decay = 1e-2
    hack_trivial = 0
    epochs = 2
    lr = 8e-4

    # Init model
    sirp = SIRNetwork(input=6, layers=4, hidden=50, output=4)

    # Init model

    source_model_name = 'i_0={}_r_0={}_p_0={}_betas={}_gammas={}.pt'.format(source_i_0_set, source_r_0_set, source_p_0_set,
                                                              source_betas,
                                                              source_gammas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(source_model_name))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sirp.parameters(), lr=lr)
        source_epochs = 2
        source_hack_trivial = 0
        source_train_size = 2000
        source_decay = 1e-2
        writer = SummaryWriter('runs/{}_scratch'.format(source_model_name))
        sirp, train_losses, run_time, optimizer = train_bundle(sirp, initial_conditions_set, t_final=t_final,
                                                               epochs=source_epochs, model_name=source_model_name,
                                                               num_batches=10, hack_trivial=source_hack_trivial,
                                                               train_size=source_train_size, optimizer=optimizer,
                                                               decay=source_decay,
                                                               writer=writer, betas=source_betas,
                                                               gammas=source_gammas)
        # Save the model
        torch.save({'model_state_dict': sirp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(source_model_name))

        # Load the checkpoint
        checkpoint = torch.load(ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(source_model_name))

    # Target model
    target_i_0_set = [0.01, 0.02]
    target_r_0_set = [0.004, 0.009]
    target_p_0_set = [0.9, 0.97]
    target_betas = [0.5, 0.7]
    target_gammas = [0.25, 0.35]

    target_model_name = 'i_0={}_r_0={}_p_0={}_betas={}_gammas={}.pt'.format(target_i_0_set, target_r_0_set, target_p_0_set,
                                                              target_betas,
                                                              target_gammas)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(target_model_name))

    except FileNotFoundError:
        print('Finetuning...')
        # Load old model
        sirp.load_state_dict(checkpoint['model_state_dict'])
        # Train
        initial_conditions_set = []
        t_0 = 0
        t_final = 20
        initial_conditions_set.append(t_0)
        initial_conditions_set.append(target_i_0_set)
        initial_conditions_set.append(target_r_0_set)
        initial_conditions_set.append(target_p_0_set)
        optimizer = torch.optim.Adam(sirp.parameters(), lr=lr)
        target_epochs = 10000
        target_hack_trivial = 0
        target_train_size = 2000
        target_decay = 1e-3
        writer = SummaryWriter('runs/{}_finetuned'.format(target_model_name))
        sirp, train_losses, run_time, optimizer = train_bundle(sirp, initial_conditions_set, t_final=t_final,
                                                               epochs=target_epochs, model_name=target_model_name,
                                                               num_batches=10, hack_trivial=target_hack_trivial,
                                                               train_size=target_train_size, optimizer=optimizer,
                                                               decay=target_decay,
                                                               writer=writer, betas=target_betas,
                                                               gammas=target_gammas)

        # Save the model
        torch.save({'model_state_dict': sirp.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(target_model_name))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIRP_bundle_total/{}'.format(target_model_name))

    # Load fine-tuned model
    sirp.load_state_dict(checkpoint['model_state_dict'])

    # Test between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)