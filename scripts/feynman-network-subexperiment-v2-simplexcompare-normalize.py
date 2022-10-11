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

import pandas as pd
from sympy import parse_expr, lambdify

import torch
import torch.nn as nn
import torch.nn.functional as F

import seml
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
ex = Experiment("feynman-network-subexperiment-v2-simplexcompare-normalize")
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

def create_normalizer(mean, std):
    """Create a function to normalize data of the form (N, d) given the means
    and stds along each of the d columns.

    Args:
        mean: (mean[1], ..., mean[d])
        std: (std[1], ..., std[d])
    """
    assert len(mean) == len(std)
    def normalize(x):
        return torch.stack([(x[:, i] - mean[i]) / std[i] for i in range(len(mean))], dim=1)
    return normalize

class ReLU2(nn.Module):
    def __init__(self):
        super(ReLU2, self).__init__()
    
    def forward(self, x):
        return torch.pow(F.relu(x, inplace=False), 2)

class ReLU3(nn.Module):
    def __init__(self):
        super(ReLU3, self).__init__()
    
    def forward(self, x):
        return torch.pow(F.relu(x, inplace=False), 3)

@ex.post_run_hook
def collect_stats(_run):
    seml.collect_exp_stats(_run)

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
    eqn = 'I.10.7'
    width = 100
    depth = 3
    lr = 1e-3
    activation = 'ReLU'
    N_TEST_POINTS = 30000
    TEST_COMPACTIFICATION = 0.8
    MAX_TRAIN_ITERS = 25000
    MAX_BATCH_SIZE = 30000
    spreadsheet = "/om2/user/ericjm/precision-ml/equations.csv"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float64

    overwrite = None
    db_collection = None
    if db_collection is not None:
        ex.observers.append(seml.create_mongodb_observer(db_collection, overwrite=overwrite))

# --------------------------
#  |-|    *
#  |-|   _    *  __
#  |-|   |  *    |/'   SEND
#  |-|   |~*~~~o~|     IT!
#  |-|   |  O o *|
# /___\  |o___O__|
# --------------------------
@ex.automain
def run(eqn, width, depth, lr, activation, 
            N_TEST_POINTS, TEST_COMPACTIFICATION, MAX_TRAIN_ITERS, MAX_BATCH_SIZE,
            spreadsheet, device, dtype, seed, _log):
    """This version gets its data not from the fixed collection of 1M points, but rather
    from the spreadsheet that describes the formula and the ranges for each variable. We
    parse the expression using SymPy and then generate new random datapoints.
    """
    torch.set_default_dtype(dtype)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    equations = pd.read_csv(spreadsheet)
    row = equations[equations['Equation'] == eqn].iloc[0]
    dimension = int(row['# variables'])
    formula = row['Formula']
    variables = [row[f'v{i}_name'] for i in range(1, dimension+1)]
    ranges = [(row[f'v{i}_low'], row[f'v{i}_high']) for i in range(1, dimension+1)]
    target = lambdify(variables, parse_expr(formula))

    ex.info['dimension'] = dimension

    TRAIN_POINTS = parameters(width, depth, dimension) // (dimension + 1)
    ex.info['TRAIN_POINTS'] = TRAIN_POINTS
    _log.debug(f"TRAIN_POINTS: {TRAIN_POINTS}")

    # create datasets
    ls = np.array([ranges[i][0] for i in range(dimension)])
    hs = np.array([ranges[i][1] for i in range(dimension)])
    xs_train = np.random.uniform(low=ls, high=hs, size=(TRAIN_POINTS, dimension))
    ys_train = target(*[xs_train[:, i] for i in range(dimension)])

    cs = (hs + ls) / 2
    ws = (hs - ls) * TEST_COMPACTIFICATION
    ls, hs = cs - ws / 2, cs + ws / 2
    xs_test = np.random.uniform(low=ls, high=hs, size=(N_TEST_POINTS, dimension))
    ys_test = target(*[xs_test[:, i] for i in range(dimension)])

    xs_train = torch.from_numpy(xs_train).to(device)
    ys_train = torch.from_numpy(ys_train).to(device).unsqueeze(dim=1)
    xs_test = torch.from_numpy(xs_test).to(device)
    ys_test = torch.from_numpy(ys_test).to(device).unsqueeze(dim=1)

    normalize = create_normalizer(xs_train.mean(dim=0), xs_train.std(dim=0))
    xs_train = normalize(xs_train)
    xs_test = normalize(xs_test)
    
    assert xs_train.dtype == dtype
    assert ys_train.dtype == dtype
    assert xs_test.dtype == dtype
    assert ys_test.dtype == dtype

    if activation == 'ReLU':
        activation_fn = nn.ReLU
    elif activation == 'Tanh':
        activation_fn = nn.Tanh
    elif activation == 'Sigmoid':
        activation_fn = nn.Sigmoid
    elif activation == 'ReLU2':
        activation_fn = ReLU2
    elif activation == 'ReLU3':
        activation_fn = ReLU3
    else:
        assert False, f"Unrecognized activation function identifier: {activation}"

    # create model
    layers = []
    for i in range(depth):
        if i == 0:
            layers.append(nn.Linear(dimension, width))
            layers.append(activation_fn())
        elif i == depth - 1:
            layers.append(nn.Linear(width, 1))
        else:
            layers.append(nn.Linear(width, width))
            layers.append(activation_fn())
    mlp = nn.Sequential(*layers).to(device)
    _log.debug("Created model.")
    _log.debug(f"Model has {sum(t.numel() for t in mlp.parameters())} parameters") 

    loss_fn = nn.MSELoss()
    optim = torch.optim.Adam(mlp.parameters(), lr=lr)

    ex.info['train'] = list()
    ex.info['test'] = list()
    min_train = float('inf')
    min_test = float('inf')
    test_at_min_train = float('inf')

    k = 0
    for step in range(MAX_TRAIN_ITERS):
        optim.zero_grad()
        if TRAIN_POINTS <= MAX_BATCH_SIZE:
            ys_pred = mlp(xs_train)
            l = loss_fn(ys_train, ys_pred)
        else:
            sample_idx = torch.arange(k, k+MAX_BATCH_SIZE, 1) % TRAIN_POINTS
            xs_batch, ys_batch = xs_train[sample_idx], ys_train[sample_idx]
            ys_pred = mlp(xs_batch)
            l = loss_fn(ys_batch, ys_pred)
            k += MAX_BATCH_SIZE
        l.backward()
        optim.step()
        with torch.no_grad():
            train_l = torch.sqrt(l).item()
            test_l = torch.sqrt(torch.mean(torch.pow(mlp(xs_test) - ys_test, 2))).item()
            if train_l < min_train:
                min_train = train_l
                test_at_min_train = test_l
            min_test = test_l if test_l < min_test else min_test
            if step % 100 == 0:
                ex.info['train'].append(train_l)
                ex.info['test'].append(test_l)
        if step % (MAX_TRAIN_ITERS // 10) == 0:
            _log.debug("{:.0f}% done with training".format(step / MAX_TRAIN_ITERS * 100))
    ex.info['min_train'] = min_train
    ex.info['min_test'] = min_test
    ex.info['test_at_min_train'] = test_at_min_train
    _log.debug("Test loss: {:.3e}".format(test_at_min_train))
