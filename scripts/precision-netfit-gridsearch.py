#!/usr/bin/env python
# coding: utf-8
"""
The purpose of this script is to do experiments where the noise isn't
in the parameter space, but rather directly added to the gradient. Is
there a phase transition there?
"""


from itertools import islice, product
import random
from re import I

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision

from scipy import optimize

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("precision-netfit-gridsearch")
ex.captured_out_filter = apply_backspaces_and_linefeeds


def dot_prod(u, v):
    assert len(u) == len(v)
    result = 0
    for i in range(len(u)):
        result += torch.sum(u[i] * v[i])
    return result.item()

def loss(param_vector, lenghts, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lenghts[i]]
        l += lenghts[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    return loss_fn(mlp(x.to(device)), y).detach().cpu().numpy()

def gradient(param_vector, lenghts, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lenghts[i]]
        l += lenghts[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    loss_fn(mlp(x.to(device)), y).backward()
    grads = []
    for param in mlp.parameters():
        grads.append(param.grad.detach().clone().cpu().numpy().flatten())
        param.grad = None
    mlp.zero_grad()
    return np.concatenate(grads)


def hvp_autograd(model, v, loss_fn, x, y):
    """Computes the Hessian-vector product Hv using torch.autograd.
    """
    def get_func(model: nn.Module, loss_fn, x, y):
        def func(*p_tensors):
            assert len(list(model.parameters())) == len(p_tensors)
            assert type(model) is nn.Sequential
            i = 0
            z = x
            for module in model:
                if type(module) is nn.Linear:
                    if module.bias is None:
                        z = nn.functional.linear(z, p_tensors[i], None)
                        i += 1
                    else:
                        z = nn.functional.linear(z, p_tensors[i], p_tensors[i+1])
                        i += 2
                if type(module) in [nn.Tanh, nn.Sigmoid, nn.ReLU]:
                    z = module(z)
            return loss_fn(z, y)
        return func
    _, Hv = torch.autograd.functional.hvp(get_func(model, loss_fn, x, y), tuple(model.parameters()), v)
    return Hv


@ex.config
def cfg():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64
    widths = [7, 10, 15, 20, 25]
    depths = [2, 3, 4]
    activations = [nn.Tanh]
    trials = 30 # the number of times to test each architecture


def train_and_observe(i, width, depth, activation, trial, device, dtype, seed):
    
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed + i)
    random.seed(seed + i)
    np.random.seed(seed + i)
    
    key = f"{width}::{depth}::{activation}::{trial}"

    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(1, width))
            layers.append(activation())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 1))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation())
    mlp = nn.Sequential(*layers).to(device)

    params = []
    shapes = []
    lenghts = []
    for param in mlp.parameters():
        param_np = param.data.detach().clone().cpu().numpy()
        shapes.append(param_np.shape)
        param_np_flat = param_np.flatten()
        lenghts.append(len(param_np_flat))
        params.append(param_np_flat)
    param_vector = np.concatenate(params)

    rmse_loss_fn_torch = lambda x, y: torch.sqrt(torch.mean(torch.pow(x-y, 2)))

    # create teacher network
    teacher_layers = []
    for i in range(depth):
        if i == 0:
            teacher_layers.append(nn.Linear(1, 5))
            teacher_layers.append(activation())
        elif i == depth - 1:
            teacher_layers.append(nn.Linear(5, 1))
        else:
            teacher_layers.append(nn.Linear(5, 5))
            teacher_layers.append(activation())
    teacher = nn.Sequential(*teacher_layers).to(device) 

    x_i = np.random.uniform(low=-1, high=1, size=(50000, 1)) 
    x_i = torch.from_numpy(x_i).to(device)
    with torch.no_grad():
        y_i = teacher(x_i)

    result = optimize.minimize(loss,
                           param_vector, 
                           args=(lenghts, shapes, mlp, rmse_loss_fn_torch, x_i, y_i, device),
                           jac=gradient,
                           method='BFGS',
                           options={
                               'disp': True,
                               'gtol': 1e-18,
                               'maxiter': 25000,
#                                'finite_diff_rel_step': 1e-15
                           },
                        )
    ex.info[key] = result.fun


@ex.automain
def run(widths, depths, activations, trials, device, dtype, seed):

    for i, (width, depth, activation, trial) in tqdm(list(enumerate(product(widths, depths, activations, range(trials))))):
        train_and_observe(i, width, depth, activation, trial, device, dtype, seed)

