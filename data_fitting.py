from tqdm import tqdm
from random import shuffle
import torch
import copy
import numpy as np
from numpy.random import uniform


def fit(model, init_bundle, betas, gammas, known_points, validation_data, epochs=100, lr=8e-4, mode='real', n_batches=3,
        susceptible_weight=1., infected_weight=1.,
        recovered_weight=1., passive_weight=1., force_init=False, verbose=False, writer=None):
    model.eval()

    # Sample randomly initial conditions, beta and gamma
    i_0 = uniform(init_bundle[1][0], init_bundle[1][1], size=1)
    r_0 = uniform(init_bundle[2][0], init_bundle[2][1], size=1)
    p_0 = uniform(init_bundle[3][0], init_bundle[3][1], size=1)
    beta = uniform(betas[0], betas[1], size=1)
    gamma = uniform(gammas[0], gammas[1], size=1)

    beta, gamma = torch.Tensor([beta]).reshape(-1, 1), torch.Tensor([gamma]).reshape(-1, 1)

    # if force_init == True fix the initial conditions as the ones given and find only the params
    if force_init:
        i_0 = known_points[0][0]
        r_0 = known_points[0][1]
        p_0 = torch.Tensor([p_0]).reshape(-1, 1)
        i_0 = torch.Tensor([i_0]).reshape(-1, 1)
        r_0 = torch.Tensor([r_0]).reshape(-1, 1)
        optimizer = torch.optim.Adam([beta, gamma, p_0], lr=lr)
    else:
        i_0 = torch.Tensor([i_0]).reshape(-1, 1)
        r_0 = torch.Tensor([r_0]).reshape(-1, 1)
        p_0 = torch.Tensor([p_0]).reshape(-1, 1)
        optimizer = torch.optim.Adam([beta, gamma, i_0, r_0, p_0], lr=lr)

    s_0 = 1 - (i_0 + r_0 + p_0)

    # Set requires_grad = True to the inputs to allow backprop
    i_0.requires_grad = True
    r_0.requires_grad = True
    p_0.requires_grad = True
    beta.requires_grad = True
    gamma.requires_grad = True

    initial_conditions = [s_0, i_0, r_0, p_0]

    known_t = copy.deepcopy(list(known_points.keys()))

    batch_size = int(len(known_t) / n_batches)

    train_losses = []
    val_losses = []
    min_val_loss = 1000

    # Iterate for epochs to find best initial conditions, beta, and gamma that optimizes the MSE/Cross Entropy between
    # my prediction and the real data
    for epoch in tqdm(range(epochs), desc='Finding the best inputs', disable=not verbose):

        # Take the time points and shuffle them
        shuffle(known_t)

        batch_filling = 0.

        epoch_loss = 0.
        batch_loss = 0.

        for idx, t in enumerate(known_t):

            batch_filling += 1

            target = known_points[t]

            t_tensor = torch.Tensor([t]).reshape(-1, 1)

            s_hat, i_hat, r_hat, p_hat = model.parametric_solution(t_tensor, initial_conditions, beta=beta, gamma=gamma)

            # If mode=='real' initial conditions for I and R are fixed
            if mode == 'real':
                i_target = target[0]
                r_target = target[1]
                loss_i = (i_target - i_hat).pow(2)
                loss_r = (r_target - r_hat).pow(2)
                loss_i = loss_i * infected_weight
                loss_r = loss_r * recovered_weight
                batch_loss += loss_i + loss_r
            else:
                s_target = target[0]
                i_target = target[1]
                r_target = target[2]
                p_target = target[3]

                loss_s = (s_target - s_hat).pow(2)
                loss_i = (i_target - i_hat).pow(2)
                loss_r = (r_target - r_hat).pow(2)
                loss_p = (p_target - p_hat).pow(2)

                loss_s = loss_s * susceptible_weight
                loss_i = loss_i * infected_weight
                loss_r = loss_r * recovered_weight
                loss_p = loss_p * passive_weight
                batch_loss += loss_s + loss_i + loss_r + loss_p

            if batch_size == batch_filling:
                # Weighting that regularizes how much we want to weight the Recovered/Susceptible curve

                batch_loss = batch_loss / len(known_points.keys())
                epoch_loss += batch_loss

                batch_loss.backward(retain_graph=True)
                optimizer.step()
                optimizer.zero_grad()

                batch_filling = 0
                batch_loss = 0.

        # For the last batch
        if batch_filling != 0:
            batch_loss = batch_loss / len(known_points.keys())
            epoch_loss += batch_loss

            batch_loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(epoch_loss)

        # To prevent r_0 to go negative in extreme cases:
        if r_0 < 0.:
            r_0 = torch.clamp(r_0, 0, 10000)

        # Adjust s_0 after update of i_0 and r_0, and update initial_conditions
        s_0 = 1 - (i_0 + r_0 + p_0)

        initial_conditions = [s_0, i_0, r_0, p_0]

        if writer:
            writer.add_scalar('Loss/train', epoch_loss, epoch)

        # Validation loss
        val_loss = 0.
        for idx, t in enumerate(validation_data):

            target = validation_data[t]

            t_tensor = torch.Tensor([t]).reshape(-1, 1)

            s_hat, i_hat, r_hat, p_hat = model.parametric_solution(t_tensor, initial_conditions, beta=beta, gamma=gamma)

            if mode == 'real':
                i_target = target[0]
                r_target = target[1]
                loss_i = (i_target - i_hat).pow(2)
                loss_r = (r_target - r_hat).pow(2)
                loss_i = loss_i * infected_weight
                loss_r = loss_r * recovered_weight
                val_loss += loss_i + loss_r
            else:
                s_target = target[0]
                i_target = target[1]
                r_target = target[2]
                p_target = target[3]

                loss_s = (s_target - s_hat).pow(2)
                loss_i = (i_target - i_hat).pow(2)
                loss_r = (r_target - r_hat).pow(2)
                loss_p = (p_target - p_hat).pow(2)

                loss_s = loss_s * susceptible_weight
                loss_i = loss_i * infected_weight
                loss_r = loss_r * recovered_weight
                loss_p = loss_p * passive_weight
                val_loss += loss_s + loss_i + loss_r + loss_p

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            optimal_i_0, optimal_r_0, optimal_p_0, optimal_beta, optimal_gamma = copy.deepcopy(i_0), copy.deepcopy(
                r_0), copy.deepcopy(p_0), copy.deepcopy(beta), copy.deepcopy(gamma),

        val_losses.append(val_loss)

    return optimal_i_0, optimal_r_0, optimal_p_0, optimal_beta, optimal_gamma, min_val_loss, val_losses


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    ce = -np.sum(targets * np.log(predictions + epsilon))
    return ce
