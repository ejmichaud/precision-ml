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

import pynvml
pynvml.nvmlInit()

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-network-experiments")
ex.captured_out_filter = apply_backspaces_and_linefeeds


def parameters(width, depth, dimension):
    """Computes number of parameters in MLP with widths `width`,
    depth `depth`, and input dimension `dimension`. Assumes the neurons
    have biases.
    """
    return (dimension * width + width) + (depth - 2)*(width * width + width) + (width + 1)

def width_given_ddp(depth, dimension, parameters):
    """Given the network depth and input dimension, computes the
    width such that the architecture (with bias) has `parameters` parameters.
    """
    if depth == 2:
        return int((parameters - 1) / (dimension + 2))
    root = (-(dimension + depth) + np.sqrt(np.power(dimension + depth, 2) - 4 * (depth - 2) * (1 - parameters))) / (2 * (depth - 2))
    return int(root)

def get_remaining_memory(i):
    """Returns memory remaining in bytes on gpu `i` (CUDA naming convention).
    """
    h = pynvml.nvmlDeviceGetHandleByIndex(i)
    info = pynvml.nvmlDeviceGetMemoryInfo(h)
    return info.free


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
    REPEATS = 3
    APPROXIMATE_TRAIN_POINTS = [4**n for n in range(3, 10)]
    N_TEST_POINTS = 30000
    TEST_COMPACTIFICATION = 0.9
    # DATA_RATIOS = [0.11, 0.33, 1.0, 3.0, 9.0] # ratio of number of datapoints to number of model parameters
    DATA_RATIOS = [1.0]
    DEPTHS = [2, 3, 4]
    LRS = [1e-2, 1e-3, 1e-4]
    # ACTIVATIONS = [nn.ReLU, nn.Tanh]
    ACTIVATIONS = [nn.ReLU]
    MAX_D = 4
    MAX_TRAIN_ITERS = 25000
    MAX_BATCH_SIZE = 30000
    MEMORY_ALLOC = 3000 # memory required to run an experiment in MegaBytes
    data_dir = "/home/eric/Downloads/Feynman_with_units"
    if torch.cuda.is_available():
        devices = [f'cuda:{i}' for i in range(torch.cuda.device_count())]
    else:
        devices = ['cpu']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

def compute_subexperiment(data, width, depth, activation, lr, dr, 
                MAX_BATCH_SIZE, MAX_TRAIN_ITERS, TEST_COMPACTIFICATION, N_TEST_POINTS,
                device, dtype, seed):
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    results = defaultdict(list)
    _, size = data.shape
    dimension = size - 1
    xs = data[:, :dimension]
    ys = data[:, dimension:]
    ls = xs.min(axis=0)
    hs = xs.max(axis=0)
    cs = (hs + ls) / 2
    ws = (hs - ls) / 2
    ls, hs = cs - ws / 2, cs + ws / 2
    interior_points = [k for k in range(len(xs)) if 
                np.all((xs[k] > TEST_COMPACTIFICATION * ls) & (xs[k] < TEST_COMPACTIFICATION * hs))]
    xs = torch.from_numpy(xs).to(device)
    ys = torch.from_numpy(ys).to(device)
    # choose test data
    test_subset = np.random.choice(interior_points, size=N_TEST_POINTS)
    xs_test = xs[test_subset, :]
    ys_test = ys[test_subset, :]
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
    # adjust to match on data ratio
    D = int(dr * D)

    # choose training data
    subset = np.random.choice([n for n in range(len(xs)) if n not in test_subset], size=D)
    xs_train = xs[subset, :]
    ys_train = ys[subset, :]

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(mlp.parameters(), lr=lr)
    
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
        with torch.no_grad():
            train_losses.append(torch.sqrt(l).item())
            test_losses.append(torch.sqrt(torch.mean(torch.pow(mlp(xs_test) - ys_test, 2))).item())
    results['train'].append(min(train_losses))
    results['test'].append(min(test_losses))
    return results

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
        APPROXIMATE_TRAIN_POINTS,
        N_TEST_POINTS,
        TEST_COMPACTIFICATION,
        DATA_RATIOS,
        DEPTHS,
        LRS,
        ACTIVATIONS, 
        MAX_D, 
        MAX_TRAIN_ITERS,
        MAX_BATCH_SIZE,
        data_dir, 
        MEMORY_ALLOC,
        devices, 
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
        if 2 <= dimension <= MAX_D:
            eqn = file.parts[-1]
            ex.info[eqn] = dict()
            ex.info[eqn]['dimension'] = dimension

            for activation, depth, D, lr, dr, trial in tqdm(list(product(ACTIVATIONS,
                                                                    DEPTHS,
                                                                    APPROXIMATE_TRAIN_POINTS,
                                                                    LRS,
                                                                    DATA_RATIOS,
                                                                    range(REPEATS) 
                                                                    )), leave=False):
                
                act_name = str(activation).split(".")[-1][:-2]
                width = width_given_ddp(depth, dimension, D)
                desc = f"width:{width};DEPTH:{depth};ACTIVATION:{act_name};LR:{lr};DATARAT:{dr}"
                ex.info[eqn][desc] = compute_subexperiment(data, width, depth, activation, lr, dr,
                    MAX_BATCH_SIZE, MAX_TRAIN_ITERS, TEST_COMPACTIFICATION, N_TEST_POINTS,
                    device, dtype, seed + trial)

