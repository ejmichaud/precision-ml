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
ex = Experiment("feynman-network-interpolation-experiments-increasingboth")
ex.captured_out_filter = apply_backspaces_and_linefeeds

# rmse_loss_fn_torch = lambda x, y: torch.sqrt(torch.mean(torch.pow(x-y, 2)))
# mse_loss_fn_torch = nn.MSELoss()

# def loss(param_vector, lengths, shapes, 
#              mlp, loss_fn, x, y, device):
#     l = 0
#     for i, param in enumerate(mlp.parameters()):
#         param_data = param_vector[l:l+lengths[i]]
#         l += lengths[i]
#         param_data_shaped = param_data.reshape(shapes[i])
#         param.data = torch.tensor(param_data_shaped).to(device)
#     return loss_fn(mlp(x.to(device)), y).detach().cpu().numpy()

# def gradient(param_vector, lengths, shapes, 
#              mlp, loss_fn, x, y, device):
#     l = 0
#     for i, param in enumerate(mlp.parameters()):
#         param_data = param_vector[l:l+lengths[i]]
#         l += lengths[i]
#         param_data_shaped = param_data.reshape(shapes[i])
#         param.data = torch.tensor(param_data_shaped).to(device)
#     loss_fn(mlp(x.to(device)), y).backward()
#     grads = []
#     for param in mlp.parameters():
#         grads.append(param.grad.detach().clone().cpu().numpy().flatten())
#         param.grad = None
#     mlp.zero_grad()
#     return np.concatenate(grads)


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
    # TRAIN_POINTS = [4**n for n in range(2, 8)]
    N_TEST_POINTS = 50000
    WIDTHS = [6, 10, 15, 30, 70, 100, 200, 400, 700]
    DEPTHS = [3]
    ACTIVATIONS = [nn.ReLU, nn.Tanh]
    MAX_D = 4
    MAX_TRAIN_ITERS = 20000
    MAX_BATCH_SIZE = 10000
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
        # TRAIN_POINTS,
        N_TEST_POINTS, 
        WIDTHS, 
        DEPTHS, 
        ACTIVATIONS, 
        MAX_D, 
        MAX_TRAIN_ITERS,
        MAX_BATCH_SIZE,
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

            for activation, depth, width, trial in tqdm(list(product(ACTIVATIONS,
                                                                    DEPTHS,
                                                                    WIDTHS,
                                                                    range(REPEATS) 
                                                                    )), leave=False):
                act_name = str(activation).split(".")[-1][:-2]
                desc = f"width:{width};DEPTH:{depth};ACTIVATION:{act_name}"
                if desc not in ex.info[eqn]:
                    ex.info[eqn][desc] = defaultdict(list)

                # create model
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

                D = sum(t.numel() for t in mlp.parameters())
                assert D == ((dimension * width + width) + (depth - 2)*(width * width + width) + (width + 1))

                # choose training data
                subset = np.random.choice([n for n in range(len(xs)) if n not in test_subset], size=D)
                xs_train = xs[subset, :]
                ys_train = ys[subset, :]

                loss_fn = nn.MSELoss()
                optim = torch.optim.Adam(mlp.parameters(), lr=1e-3)
                
                train_losses = []
                test_losses = []

                k = 0
                for _ in range(MAX_TRAIN_ITERS):
                    optim.zero_grad()
                    if D <= MAX_BATCH_SIZE:
                        ys_pred = mlp(xs_train)
                        l = loss_fn(ys_train, ys_pred)
                    else:
                        sample_idx = torch.arange(k, k+MAX_BATCH_SIZE, 1) % D
                        xs_batch, ys_batch = xs_train[sample_idx], ys_train[sample_idx]
                        ys_pred = mlp(xs_batch)
                        l = loss_fn(ys_batch, ys_pred)
                        k += MAX_BATCH_SIZE
                    l.backward()
                    optim.step()
                    train_losses.append(l.item())
                    with torch.no_grad():
                        test_losses.append(loss_fn(ys_test, mlp(xs_test)).item())
                
                ex.info[eqn][desc]['train'].append(min(train_losses))
                ex.info[eqn][desc]['test'].append(min(test_losses))

