#!/usr/bin/env python
# coding: utf-8
"""
The purpose of this script is to fit some of the AI Feynman datasets using
0th and 1st order interpolation. We only fit problems below a certain dimension
since computing the simplices gets hard in more dimensions!
"""

from collections import defaultdict
from itertools import islice, product
import random
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from scipy import optimize

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-network-interpolation-experiments")
ex.captured_out_filter = apply_backspaces_and_linefeeds

rmse_loss_fn_torch = lambda x, y: torch.sqrt(torch.mean(torch.pow(x-y, 2)))
mse_loss_fn_torch = nn.MSELoss()

def loss(param_vector, lengths, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lengths[i]]
        l += lengths[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    return loss_fn(mlp(x.to(device)), y).detach().cpu().numpy()

def gradient(param_vector, lengths, shapes, 
             mlp, loss_fn, x, y, device):
    l = 0
    for i, param in enumerate(mlp.parameters()):
        param_data = param_vector[l:l+lengths[i]]
        l += lengths[i]
        param_data_shaped = param_data.reshape(shapes[i])
        param.data = torch.tensor(param_data_shaped).to(device)
    loss_fn(mlp(x.to(device)), y).backward()
    grads = []
    for param in mlp.parameters():
        grads.append(param.grad.detach().clone().cpu().numpy().flatten())
        param.grad = None
    mlp.zero_grad()
    return np.concatenate(grads)


# --------------------------
#    ,-------------.
#   (_\  CONFIG     \
#      |    OF      |
#      |    THE     |
#     _| EXPERIMENT |
#    (_/_____(*)___/
#             \\
#              ))
#              ^
# --------------------------
@ex.config
def cfg():
    REPEATS = 5
    TRAIN_POINTS = [4**n for n in range(2, 8)]
    N_TEST_POINTS = 50000
    WIDTHS = [5, 10, 20, 40, 80]
    DEPTHS = [3]
    ACTIVATIONS = [nn.ReLU, nn.Tanh]
    MAX_D = 5
    MAX_TRAIN_ITERS = 5000
    data_dir = "/home/eric/Downloads/Feynman_with_units"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64


# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(REPEATS, 
        TRAIN_POINTS,
        N_TEST_POINTS, 
        WIDTHS, 
        DEPTHS, 
        ACTIVATIONS, 
        MAX_D, 
        MAX_TRAIN_ITERS,
        data_dir, 
        device, 
        dtype, 
        seed):

    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    for i, file in tqdm(list(enumerate(Path(data_dir).iterdir()))):
        data = np.loadtxt(file, dtype=np.float64)
        _, size = data.shape
        dimension = size - 1
        if dimension <= MAX_D:
            xs = torch.from_numpy(data[:, :dimension]).to(device)
            ys = torch.from_numpy(data[:, dimension:]).to(device)
            test_subset = np.random.choice(list(range(len(xs))), size=N_TEST_POINTS)
            xs_test = xs[test_subset, :]
            ys_test = ys[test_subset, :]

            eqn = file.parts[-1]
            ex.info[eqn] = dict()
            ex.info[eqn]['dimension'] = dimension
            for width, depth, activation, D, trial in tqdm(list(product(WIDTHS,
                                                                    DEPTHS, 
                                                                    ACTIVATIONS, 
                                                                    TRAIN_POINTS,
                                                                    range(REPEATS))), leave=False):
                act_name = str(activation).split(".")[-1][:-2]
                desc = f"width:{width};DEPTH:{depth};ACTIVATION:{act_name};TRAINPOINTS:{D}"
                if desc not in ex.info[eqn]:
                    ex.info[eqn][desc] = defaultdict(list)
                # choose training data
                subset = np.random.choice([n for n in range(len(xs)) if n not in test_subset], size=D)
                xs_train = xs[subset, :]
                ys_train = ys[subset, :]

                layers = []
                for i in range(depth):
                    if i == 0:
                        layers.append(nn.Linear(dimension, width))
                        layers.append(activation())
                    elif i == depth - 1:
                        layers.append(nn.Linear(width, 1))
                    else:
                        layers.append(nn.Linear(width, width))
                        layers.append(activation())
                mlp = nn.Sequential(*layers).to(device)

                params = []
                shapes = []
                lengths = []
                for param in mlp.parameters():
                    param_np = param.data.detach().clone().cpu().numpy()
                    shapes.append(param_np.shape)
                    param_np_flat = param_np.flatten()
                    lengths.append(len(param_np_flat))
                    params.append(param_np_flat)

                param_vector = np.concatenate(params)

                result = optimize.minimize(loss,
                                    param_vector, 
                                    args=(lengths, shapes, mlp, mse_loss_fn_torch, xs_train, ys_train, device),
                                    jac=gradient,
                                    method='BFGS',
                                    options={
                                        'disp': False,
                                        'gtol': 1e-40,
                                        'maxiter': MAX_TRAIN_ITERS,
                                    })

                l = 0
                for j, param in enumerate(mlp.parameters()):
                    param_data = result.x[l:l+lengths[j]]
                    l += lengths[j]
                    param_data_shaped = param_data.reshape(shapes[j])
                    param.data = torch.tensor(param_data_shaped).to(device)

                with torch.no_grad():
                    train_loss = rmse_loss_fn_torch(mlp(xs_train), ys_train).item()
                    test_loss = rmse_loss_fn_torch(mlp(xs_test), ys_test).item()
                    ex.info[eqn][desc]['train'].append(train_loss)
                    ex.info[eqn][desc]['test'].append(test_loss)


