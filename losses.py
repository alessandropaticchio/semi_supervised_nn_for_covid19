import torch
from torch.autograd import grad
from constants import device, dtype

def sirp_loss(t, s, i, r, p, beta, gamma, decay=0):
    s_prime = dfx(t, s)
    i_prime = dfx(t, i)
    r_prime = dfx(t, r)
    p_prime = dfx(t, p)

    N = s + i + r

    loss_s = s_prime + (beta * i * s) / N
    loss_i = i_prime - (beta * i * s) / N + gamma * i
    loss_r = r_prime - gamma * i
    loss_p = p_prime

    # Regularize to give more importance to initial points
    loss_s = loss_s * torch.exp(-decay * t)
    loss_i = loss_i * torch.exp(-decay * t)
    loss_r = loss_r * torch.exp(-decay * t)
    loss_p = loss_p * torch.exp(-decay * t)

    loss_s = (loss_s.pow(2)).mean()
    loss_i = (loss_i.pow(2)).mean()
    loss_r = (loss_r.pow(2)).mean()
    loss_p = (loss_p.pow(2)).mean()

    total_loss = loss_s + loss_i + loss_r + loss_p

    return total_loss


def mse_loss(known, model, initial_conditions):
    mse_loss = 0.
    for t in known.keys():
        t_tensor = torch.Tensor([t]).reshape(-1, 1)
        s_hat, i_hat, r_hat = model.parametric_solution(t_tensor, initial_conditions)
        loss_s = (known[t][0] - s_hat).pow(2)
        loss_i = (known[t][1] - i_hat).pow(2)
        loss_r = (known[t][2] - i_hat).pow(2)

        mse_loss += loss_s + loss_i + loss_r
    return mse_loss


def trivial_loss(infected, hack_trivial):
    trivial_loss = 0.

    for i in infected:
        trivial_loss += i

    trivial_loss = hack_trivial * torch.exp(- (trivial_loss) ** 2)
    return trivial_loss

def dfx(x, f):
    # Calculate the derivative with auto-differentiation
    x = x.to(device)
    grad_outputs = torch.ones(x.shape, dtype=dtype)
    grad_outputs = grad_outputs.to(device)

    return grad([f], [x], grad_outputs=grad_outputs, create_graph=True)[0]


