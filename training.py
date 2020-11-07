import numpy as np
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from losses import *
from constants import device, ROOT_DIR
from utils import generate_dataloader, generate_grid
from numpy.random import uniform


def train_bundle(model, initial_conditions_set, t_final, epochs, train_size, optimizer, betas, gammas, model_name,
                 decay=0, num_batches=1, hack_trivial=False,
                 treshold=float('-inf'), verbose=True, writer=None):

    # Train mode
    model.train()

    # Initialize model to return
    best_model = model

    # Fetch parameters of the differential equation
    t_0, initial_conditions_set = initial_conditions_set[0], initial_conditions_set[1:]

    # Initialize losses arrays
    train_losses, val_losses, min_loss = [], [], 1

    # Points selection
    grid = generate_grid(t_0, t_final, train_size)

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc='Training', disable=not verbose):
        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0, t_final, batch_size, perturb=True)

        train_epoch_loss = 0.0

        for i, t in enumerate(t_dataloader, 0):
            # Sample randomly initial conditions, beta and gamma
            i_0 = uniform(initial_conditions_set[0][0], initial_conditions_set[0][1], size=batch_size)
            r_0 = uniform(initial_conditions_set[1][0], initial_conditions_set[1][1], size=batch_size)
            beta = uniform(betas[0], betas[1], size=batch_size)
            gamma = uniform(gammas[0], gammas[1], size=batch_size)

            i_0 = torch.Tensor([i_0]).reshape((-1, 1))
            r_0 = torch.Tensor([r_0]).reshape((-1, 1))
            beta = torch.Tensor([beta]).reshape((-1, 1))
            gamma = torch.Tensor([gamma]).reshape((-1, 1))

            s_0 = 1 - (i_0 + r_0)
            initial_conditions = [s_0, i_0, r_0]

            #  Network solutions
            s, i, r = model.parametric_solution(t, initial_conditions, beta, gamma, mode='bundle_total')

            # Loss computation
            batch_loss = sir_loss(t, s, i, r, beta=beta, gamma=gamma, decay=decay)

            # Hack to prevent the network from solving the equations trivially
            if hack_trivial:
                batch_trivial_loss = trivial_loss(i, hack_trivial)
                batch_loss = batch_loss + batch_trivial_loss

            # Optimization
            batch_loss.backward()
            optimizer.step()
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # Keep the loss function history
        train_losses.append(train_epoch_loss)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/Log-train', np.log(train_epoch_loss), epoch)

        # Keep the best model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

        # If a treshold is passed, we stop training when it is reached. Notice default value is -inf
        if train_epoch_loss < treshold:
            break

        # Backup save
        if epoch % 500 == 0 and epoch != 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       ROOT_DIR + '/models/SIR_bundle_total/{}'.format(model_name))

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer
